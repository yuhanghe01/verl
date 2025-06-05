"""
FSDP wrapper for SFT workers
"""
import logging
import os
import warnings
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from tensordict import TensorDict
from peft import TaskType, get_peft_model, LoraConfig
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.py_functional import convert_to_regular_types
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available
from torch import optim
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq
from verl.utils.model import get_generation_config, print_model_size, update_model_config
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), 
                                       mesh_dim_names=["ddp", "fsdp"])
    return device_mesh

def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    
    return sharding_strategy

# STF large model worker
class SFTLMWorker(Worker):
    def __init__(self, config, role: str):
        super().__init__()
        self.config = config
        self.role = role

        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl", 
                                                 rank=rank, world_size=world_size)

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(world_size=world_size, 
                                              fsdp_size=self.config.model.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(device_name, 
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size), 
                                                        mesh_dim_names=["dp", "sp"])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self._is_offload_param = self.config.model.fsdp_config.get("param_offload", False)
        self._is_offload_optimizer = self.config.model.fsdp_config.get("optimizer_offload", False)


    def create_sft_dataset(self, data_paths, data_config, tokenizer):
        """Create a dataset."""
        if data_config.custom_cls.get("path", None):
            from verl.utils.import_utils import load_extern_type
            dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Then check if multi-turn dataset should be used
        elif data_config.get("multiturn", {}).get("enable", False):
            dataset_cls = MultiTurnSFTDataset
        else:
            dataset_cls = SFTDataset

        dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)

        return dataset

    def get_sampler(self, dataset):
        if self.config.data.traindata_shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=dataset)

        return sampler

    def _create_dataloader(self):
        train_dataset = self.create_sft_dataset(
            data_paths=self.config.data.train_files,
            data_config=self.config.data,
            tokenizer=self.tokenizer,
        )
        val_dataset = self.create_sft_dataset(
            data_paths=self.config.data.val_files,
            data_config=self.config.data,
            tokenizer=self.tokenizer,
        )

        train_sampler = self.get_sampler(train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=self.config.data.dataloader_num_workers,
            drop_last=True,
            collate_fn=None,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.dataloader_num_workers,
            shuffle=False,
            drop_last=False,
            collate_fn=None,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        self.total_training_steps = total_training_steps
        self.one_epoch_steps = len(self.train_dataloader)

    def _build_model_optimizer(self):
        trust_remote_code = self.config.model.trust_remote_code
        model_path = self.config.model.model_path
        local_model_path = copy_to_local(model_path)

        # self.tokenizer = hf_tokenizer(local_model_path,
        #                               trust_remote_code=trust_remote_code)

        log_gpu_memory_usage("before model allocation", logger=logger)

        self.processor = hf_processor(local_model_path, 
                                      trust_remote_code=trust_remote_code)

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        model_config = AutoConfig.from_pretrained(local_model_path, 
                                                  trust_remote_code=trust_remote_code)
        self.model_config = model_config

        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.config.model.use_remove_padding, "Ulysses Sequence Parallel requires use_remove_padding to be True"

        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings, 
                                                       mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
                module_class = AutoModelForVision2Seq
            else:
                module_class = AutoModelForCausalLM

            model = module_class.from_pretrained(
                pretrained_model_name_or_path=local_model_path,
                torch_dtype=torch_dtype,
                config=model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if self.config.model.use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=model)

            if self.config.model.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                apply_monkey_patch(
                    model=model,
                    use_remove_padding=self.config.model.use_remove_padding,
                    ulysses_sp_size=self.config.ulysses_sequence_parallel_size,
                    use_fused_kernels=self.config.model.use_fused_kernels,
                )

            if self.config.model.get('lora_rank', 0) > 0:
                model.enable_input_require_grads()
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.get('lora_rank', 0),
                    'lora_alpha': self.config.model.get('lora_alpha', 32),
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': 'none'
                }
                model = get_peft_model(model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        # TODO: check if this line is needed
        torch.distributed.barrier()

        if self.device_mesh.get_rank() == 0:
            print_model_size(model)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(module=model,
                                                config=self.config.model.fsdp_config.wrap_policy,
                                                is_lora=self.config.model.get('lora_rank', 0) > 0)
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)
            self._is_offload_param = False
            self._is_offload_optimizer = False

        fsdp_strategy = self.config.model.fsdp_config.fsdp_strategy
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        if fsdp_strategy == "fsdp":
            model = FSDP(
                model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, 
                                             reduce_dtype=torch.float32,
                                             cast_forward_inputs=True)

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": self.config.model.fsdp_config.reshard_after_forward,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, fsdp_mesh, cpu_offload)
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # if enable_activation_offload:
        #     enable_activation_offloading(model_fsdp, fsdp_strategy, enable_gradient_checkpointing)

        log_gpu_memory_usage(f"After FSDP init", logger=logger)

        model_optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.get("betas", (0.9, 0.999)),
            weight_decay=self.config.optim.get("weight_decay", 1e-2),
        )

        if self.config.optim.lr_scheduler == 'cosine':
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer=model_optimizer,
                                                           num_warmup_steps=0,
                                                           num_training_steps=self.total_training_steps,
                                                           )
        elif self.config.optim.lr_scheduler == 'wsd':
            lr_scheduler = get_wsd_schedule_with_warmup(optimizer=model_optimizer,
                                                        num_warmup_steps=0,
                                                        num_training_steps=self.total_training_steps,
                                                        )
        else:
            raise NotImplementedError(f"not implement {self.config.optim.lr_scheduler}")


        self.flops_counter = FlopsCounter(model_config)
        self.checkpoint_manager = FSDPCheckpointManager(
            model = model,
            optimizer = model_optimizer,
            lr_scheduler = lr_scheduler,
            processing_class= self.processor if self.processor is not None else self.tokenizer,
            checkpoint_contents = self.config.model.checkpoint.contents,
        )

        return model, model_optimizer, lr_scheduler, model_config
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        model_path = self.config.model.model_path
        local_model_path = copy_to_local(model_path)

        self.tokenizer = hf_tokenizer(local_model_path,
                                      trust_remote_code=True)
        self._create_dataloader()
        #step 1: initialize model, optimizer and lr_scheduler
        self.fsdp_model, self.model_optimizer, self.model_lr_scheduler, self.model_config = self._build_model_optimizer()
        #get the original unwrapped module
        if fsdp_version(self.fsdp_model) == 1:
            self.model = self.fsdp_model._fsdp_wrapped_module

        # self._is_offload_param =False
        if self._is_offload_param:
            # offload the model to CPU
            offload_fsdp_model_to_cpu(self.fsdp_model)
            log_gpu_memory_usage(f"After offload fsdp model during init", logger=logger)

        # self._is_offload_optimizer = False
        if self._is_offload_optimizer:
            # offload the optimizer to CPU
            load_fsdp_optimizer(optimizer=self.model_optimizer)
            log_gpu_memory_usage(f"After offload fsdp optimizer during init", logger=logger)

        # step 3: initialize dataloader

    def save_checkpoint(self):
        # if self._is_offload_param:
        #     load_fsdp_model_to_gpu(self.module_fsdp)

        local_path = os.path.join(self.config.trainer.ckpt_save_dir, f"global_step_{self.global_steps}")
        os.makedirs(local_path, exist_ok=True)    
        self.checkpoint_manager.save_checkpoint(local_path=local_path, 
                                                hdfs_path=None, 
                                                global_step=self.global_steps, 
                                                max_ckpt_to_keep=self.config.trainer.max_ckpt_to_keep)

        torch.distributed.barrier()

        # if self._is_offload_param:
        #     offload_fsdp_model_to_cpu(self.module_fsdp)

    def run_inference_gsm8k(self):
        """Compute loss with optional sequence parallelism and remove padding features"""

        use_sp = self.config.model.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        self.fsdp_model.eval()
        accu_num, total_num = 0, 0
        for batch_id, batch in enumerate(self.val_dataloader):
            # print('eval {}/{}'.format(batch_id, len(self.val_dataloader)))
            # Move inputs to GPU and prepare loss mask
            input_ids = batch["input_ids"].to(device_name)
            attention_mask = batch["attention_mask"].to(device_name)
            position_ids = batch["position_ids"].to(device_name)

            # Context manager for sequence parallel if needed
            context = self.sharding_manager if use_sp else nullcontext()
            with context, torch.autocast(device_type=device_name, dtype=torch.bfloat16):
                if not use_sp:
                    # Standard forward pass without sequence parallel
                    labels = input_ids[:, 1:].contiguous()
                    with torch.no_grad():
                        output = self.fsdp_model(input_ids=input_ids, 
                                                attention_mask=attention_mask, 
                                                position_ids=position_ids, 
                                                use_cache=False)
                    logits = output.logits

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels.contiguous()
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.fsdp_model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)

                    pred_label = torch.argmax(torch.nn.functional.softmax(shift_logits, dim=-1), dim=-1)

                    accu_num_tmp = torch.sum((pred_label == shift_labels).to(torch.float32)).item()
                    total_num_tmp = shift_labels.numel()

                    accu_num += accu_num_tmp
                    total_num += total_num_tmp
                else:
                    # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                    # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                    # 1. All SP ranks will receive the *SAME* batch
                    # 2. Different SP groups will receive *DIFFERENT* batches
                    # This is implemented by the DistributedSampler
                    batch_size, seqlen = input_ids.shape
                    # Remove padding
                    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                    # Unpad position_ids to align rotary
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                    # Pad and slice inputs for sequence parallelism
                    input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                    # For computing loss
                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                    # Forward pass
                    self.fsdp_model.eval()
                    with torch.no_grad():
                        output = self.fsdp_model(
                            input_ids=input_ids_rmpad_sliced,
                            attention_mask=None,  # Not needed with flash attention varlen
                            position_ids=position_ids_rmpad_padded,
                            use_cache=False,
                        )

                    # Compute loss locally then aggregate
                    logits_rmpad = output.logits.squeeze(0)
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)

                    pred_label = torch.argmax(torch.nn.functional.softmax(logits_rmpad, dim=-1), dim=-1)
                    accu_num_tmp += torch.sum((pred_label == input_ids_rmpad_rolled).to(torch.float32)).item()
                    total_num_tmp = input_ids_rmpad_rolled.numel()

                    accu_num += accu_num_tmp
                    total_num += total_num_tmp
        accu_rate = accu_num / (total_num + 1e-8)
        print(f"Validation Accuracy: {accu_rate:.4f}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.config.model.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(device_name)
        attention_mask = batch["attention_mask"].to(device_name)
        position_ids = batch["position_ids"].to(device_name)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.fsdp_model.train()
        context = self.sharding_manager if use_sp else nullcontext()
        # with context, torch.autocast(device_type=device_name, dtype=torch.bfloat16):
        with context, torch.autocast(device_type=device_name, dtype=torch.float16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.fsdp_model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                loss = loss * loss_mask.to(loss.device)
            else:
                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            if do_backward:
                loss.backward()

            return loss

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()
        log_gpu_memory_usage("Before model_optimizer zero_grad", logger=logger)
        self.model_optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch, do_backward=True) / n_micro_batches
            step_loss += loss.item()

        grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.model_optimizer.zero_grad()
        else:
            self.model_optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        # self.model_lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.model_lr_scheduler.get_last_lr()[0]

        step_loss = torch.tensor(step_loss).to(device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        if self.device_mesh.get_rank() == 0:
            logger.info(f"Step {self.global_steps}, Loss: {step_loss.item():.4f}, LR: {lr:.6f}")

        return step_loss.item()

    def load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.fsdp_model)

        self.checkpoint_manager.load_checkpoint(local_path=local_path, 
                                                hdfs_path=hdfs_path, 
                                                del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.fsdp_model)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.model_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def launch_train(self):
        self.global_steps = 0
        progress_bar = tqdm(total=self.total_training_steps,
                    initial=self.global_steps, 
                    desc="Training Progress",)
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_id, batch in enumerate(self.train_dataloader):
                if not isinstance(batch, TensorDict):
                    batch = TensorDict(batch, batch_size=[self.config.data.train_batch_size])
                batch = batch.to(device_name)
                self.training_step(batch)
                self.global_steps += 1
                if batch_id % 10 == 0:
                    print(f"Epoch {epoch}, Step {self.global_steps}, Loss: {self.training_step(batch):.4f}, LR: {self.model_lr_scheduler.get_last_lr()[0]:.5f}")
                    progress_bar.update(10)
            
            if epoch % self.config.trainer.eval_every_n_epochs == 0 and epoch > 0:
                self.run_inference_gsm8k()
            
            if epoch % self.config.trainer.save_every_n_epochs == 0 and epoch > 0:
                self.save_checkpoint()

            self.model_lr_scheduler.step()

        progress_bar.close()
        logger.info("Training completed.")