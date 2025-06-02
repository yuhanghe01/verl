import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
import hydra
import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import RandomSampler, SequentialSampler, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.trainer.sft.ray_trainer import RaySFTTrainer
from verl.trainer.sft.ray_trainer import ResourcePoolManager
from verl.utils import hf_tokenizer
from verl.utils.debug import log_gpu_memory_usage

if is_cuda_available:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import pad_input, unpad_input, rearrange, index_first_axis

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))

def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None

def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import DictConfig, ListConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    
    return obj

class FSDPSFTTrainer:
    def __init__(self, 
                 config,
                 device_mesh,
                 ulysses_device_mesh,
                 tokenizer, 
                 train_dataset: Dataset, 
                 val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        #self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        # self._normalize_config_bsz()

        # Set sequence parallel size
        # self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        # if self.device_mesh.get_rank() == 0:
        #     print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
        #     print(f"Using remove padding: {self.use_remove_padding}")
        # build model
        self.get_model_and_optimizer()
        self.get_lr_scheduler()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # build resource pool manager
        global_pool_id = 'global_pool'
        resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec)

        self.device_name = get_device_name()

        #get the data sampler
        self.sampler = self.get_sampler(self.config.data, self.train_dataset)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def get_model_and_optimizer(self):
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        # init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)

        # with init_context():
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            config=config,
            torch_dtype=torch.float32,
            attn_implementation="flash_attention_2",
            trust_remote_code=trust_remote_code,
        )

        if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch

            apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

        # Apply Liger kernel if use_liger is enabled
        if self.config.model.get("use_liger", False):
            from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=self.model)

        if self.config.model.get("lora_rank", 0) > 0:
            self.model.enable_input_require_grads()
            # Convert config to regular Python types before creating PEFT model
            lora_config = {
                "task_type": TaskType.CAUSAL_LM,
                "r": self.config.model.lora_rank,
                "lora_alpha": self.config.model.lora_alpha,
                "target_modules": convert_to_regular_types(self.config.model.target_modules),
                "bias": "none",
            }
            self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        # if self.device_mesh.get_rank() == 0:
        #     print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(
            module=self.model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            device_id=get_torch_device().current_device(),
            cpu_offload=cpu_offload,
            use_orig_params=False,
        )

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        # self.steps_per_epoch = len(self.train_dataloader)
        # self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        # if self.device_mesh.get_rank() == 0:
        #     print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

        return None
        # num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        # if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
        #     self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        # elif self.config.optim.lr_scheduler == "wsd":
        #     self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        # else:
        #     raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")
        
    def get_lr_scheduler(self):
        """Get the learning rate scheduler."""
        # num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
        num_warmup_steps = 100
        self.total_steps = 100000

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")
        
    def get_sampler(self, data_config, dataset):
        if data_config.traindata_shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(data_config.seed)
            sampler = RandomSampler(dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=dataset)

        return sampler

    def run(self):
        #get everything ready for Ray SFT Trainer
        self.get_model_and_optimizer()
        self.get_lr_scheduler()

        trainer = RaySFTTrainer(
            config=self.config,
            fsdp_model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            tokenizer=self.tokenizer,
            resource_pool_manager=self.resource_pool_manager,
            train_dataset=self.train_dataset,
            val_dataset=self.val_dataset,
            train_sampler=self.sampler)
        # Initialize the Ray SFT Trainer
        # ray.init()
        trainer.init_workers()
        trainer.fit()

@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, 
                                   mesh_shape=(world_size,), 
                                   mesh_dim_names=("fsdp",))
    
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type=device_name, 
                                           mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), 
                                           mesh_dim_names=("dp", "sp"))
    
    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTTrainer(config=config, 
                             device_mesh=device_mesh,
                             ulysses_device_mesh=ulysses_device_mesh,
                             tokenizer=tokenizer, 
                             train_dataset=train_dataset, 
                             val_dataset=val_dataset)

    trainer.run()


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()