"""
FSDP wrapper for SFT workers
"""
import logging
import os
import functools
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from tensordict import TensorDict
from verl.single_controller.base import Worker
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.utils.data import RandomSampler, SequentialSampler
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available
from torch import optim
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq
from verl.utils.model import get_generation_config, print_model_size, update_model_config
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs, get_ulysses_sequence_parallel_world_size
from omegaconf import OmegaConf
from verl.utils.tracking import Tracking

from ray.train.torch import prepare_model, prepare_data_loader
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from typing import Set, Type

logger = logging.getLogger(__file__)

device_name = get_device_name()

def extract_transformer_block_classes(model: torch.nn.Module):
    class_counter = {}
    for _, module in model.named_modules():
        cls = type(module)
        class_counter[cls] = class_counter.get(cls, 0) + 1
        
    return {cls for cls, count in class_counter.items() if count > 1}

from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    input_ids, labels = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return input_ids, labels

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
class SFTLMWorker:
    def __init__(self, config):
        self.config = config
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # if not dist.is_initialized():
        #     rank = int(os.environ.get("RANK", 0))
        #     world_size = int(os.environ.get("WORLD_SIZE", 1))
        #     dist.init_process_group(backend="cpu:gloo,cuda:nccl" if is_cuda_available else "cpu:gloo,npu:hccl",
        #                              rank=rank, world_size=world_size)

        # build device mesh for FSDP
        world_size = dist.get_world_size()
        # self.device_mesh = create_device_mesh(world_size=world_size, 
        #                                       fsdp_size=self.config.model.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        # self.ulysses_device_mesh = None
        # self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        # dp = world_size // self.ulysses_sequence_parallel_size
        # if self.ulysses_sequence_parallel_size > 1:
        #     self.ulysses_device_mesh = init_device_mesh(device_name, 
        #                                                 mesh_shape=(dp, self.ulysses_sequence_parallel_size), 
        #                                                 mesh_dim_names=["dp", "sp"])

        # self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # self._is_offload_param = self.config.model.fsdp_config.get("param_offload", False)
        # self._is_offload_optimizer = self.config.model.fsdp_config.get("optimizer_offload", False)
        
        logger.info('initialize model and optimizer')
        self.init_model()
        logger.info('create dataloader')
        self._create_dataloader()

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
        self.train_dataloader = prepare_data_loader(self.train_dataloader)
        self.val_dataloader = prepare_data_loader(self.val_dataloader)

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        self.total_training_steps = total_training_steps
        self.one_epoch_steps = len(self.train_dataloader)

    def _build_model_optimizer(self):
        trust_remote_code = self.config.model.trust_remote_code
        model_path = self.config.model.model_path
        local_model_path = copy_to_local(model_path)
        log_gpu_memory_usage("before model allocation", logger=logger)

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        model_config = AutoConfig.from_pretrained(local_model_path, 
                                                  trust_remote_code=trust_remote_code)
        self.model_config = model_config
        
        
        model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_model_path,
                torch_dtype=torch.float32,
                config=model_config,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
        )
        
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16)
        
        model_optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.get("betas", (0.9, 0.999)),
            weight_decay=self.config.optim.get("weight_decay", 1e-2),
        )
        
        transformer_auto_wrapper_policy = functools.partial(
            transformer_auto_wrap_policy,
            module = model,
            transformer_layer_cls=extract_transformer_block_classes(model),
        )

        # model = FSDP(
        #     model,
        #     # device_id=self.device,
        #     cpu_offload=CPUOffload(offload_params=True),
        #     sharding_strategy=ShardingStrategy.FULL_SHARD,
        #     mixed_precision=mp_policy,
        # )
        
        # model = FSDP(
        #     model,
        #     device_id=self.device,
        #     cpu_offload=CPUOffload(offload_params=False),
        #     auto_wrap_policy=transformer_auto_wrapper_policy,
        #     sharding_strategy=ShardingStrategy.FULL_SHARD,
        #     mixed_precision=mp_policy,
        #     sync_module_states=True,
        #     forward_prefetch=True,
        #     limit_all_gathers=True,
        #     use_orig_params=False,
        # )
        
        
        
        # auto_wrap_policy = transformer_auto_wrap_policy(module_classes=block_classes)

        # Wrap model with FSDP
        
        transformer_auto_wrapper_policy = functools.partial(
            transformer_auto_wrap_policy,
            module = model,
            transformer_layer_cls=extract_transformer_block_classes(model),
        )
        model = FSDP(
            model,
            # device_id=torch.cuda.current_device(),
            cpu_offload=CPUOffload(offload_params=True),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=transformer_auto_wrapper_policy,
            mixed_precision=MixedPrecision(param_dtype=torch.float16, 
                                           reduce_dtype=torch.float16, 
                                           buffer_dtype=torch.float16),
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
            
        return model, model_optimizer, lr_scheduler, model_config
    
    def init_model(self):
        model_path = self.config.model.model_path
        local_model_path = copy_to_local(model_path)

        self.tokenizer = hf_tokenizer(local_model_path,
                                      trust_remote_code=self.config.model.trust_remote_code,)
        self._create_dataloader()
        #step 1: initialize model, optimizer and lr_scheduler
        self.fsdp_model, self.model_optimizer, self.model_lr_scheduler, self.model_config = self._build_model_optimizer()
        self.vocab_size = self.fsdp_model.config.vocab_size
        
        return None

    def save_checkpoint(self):
        local_path = os.path.join(self.config.trainer.ckpt_save_dir, f"global_step_{self.global_steps}")
        os.makedirs(local_path, exist_ok=True)
        if self.config.model.fsdp_config.fsdp_strategy in ["fsdp"]:
            #FSDP1 checkpoint saving
            from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType
            with FSDP.state_dict_type(
                self.fsdp_model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            ):
                full_state_dict = self.fsdp_model.state_dict()
            if dist.get_rank() == 0:
                # Save the model and tokenizer
                self.fsdp_model.save_pretrained(local_path, state_dict=full_state_dict, safe_serialization=True)
                self.tokenizer.save_pretrained(local_path)
        elif self.config.model.fsdp_config.fsdp_strategy == "fsdp2":
            fsdp_state_dict = self.fsdp_model.state_dict()
            self.fsdp_model.save_pretrained(local_path, state_dict=fsdp_state_dict)
        else:
            raise NotImplementedError(f"not implement {self.config.model.fsdp_config.fsdp_strategy}")

    def run_inference_gsm8k(self):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.config.model.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        self.fsdp_model.eval()
        accu_num, total_num = 0, 0
        for batch_id, batch in enumerate(self.val_dataloader):
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

        return accu_rate

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.config.model.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        input_ids = batch["input_ids"].to(device_name)
        attention_mask = batch["attention_mask"].to(device_name)
        position_ids = batch["position_ids"].to(device_name)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        
        labels = input_ids[:, 1:].contiguous()
        output = self.fsdp_model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)
        logits = output.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels.contiguous()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        loss = loss * loss_mask.to(loss.device)
        loss = torch.sum(loss) / (loss_mask.sum() + 1e-8)
        if do_backward:
            loss.backward()

        return loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()
        self.model_optimizer.zero_grad()
        loss = self._compute_loss_and_backward(batch=batch, do_backward=True)
        if self.config.model.fsdp_config.fsdp_strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
            if not torch.isfinite(grad_norm):
                logger.info(f"WARN: grad_norm is not finite: {grad_norm}")
                self.model_optimizer.zero_grad()
            else:
                self.model_optimizer.step()
        else:
            self.model_optimizer.step()
            
        return loss

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
    
    def fit(self):
        track_logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.global_steps = 0
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_id, batch in enumerate(self.train_dataloader):
                if not isinstance(batch, TensorDict):
                    batch = TensorDict(batch, batch_size=[self.config.data.train_batch_size])
                batch = batch.to(device_name)
                loss_val = self.training_step(batch)
                self.global_steps += 1
                if batch_id % 10 == 0:
                    log_info = {'train/epoch': epoch,
                                'train/step': int(self.global_steps),
                                'train/total_steps': self.total_training_steps,
                                'train/loss': loss_val.item(),
                                'train/lr(x1000)': self.model_lr_scheduler.get_last_lr()[0]*1000}
                    track_logger.log(log_info, step=self.global_steps)

            if epoch % self.config.trainer.eval_every_n_epochs == 0 and epoch >= 0 and self.config.trainer.run_evaluation:
                accu_rate = self.run_inference_gsm8k()
                log_info = {'eval/epoch': epoch,
                            'eval/accu_rate': accu_rate}
                track_logger.log(log_info, step=self.global_steps)
            
            if epoch % self.config.trainer.save_every_n_epochs == 0 and epoch > 0:
                self.save_checkpoint()
                
            self.model_lr_scheduler.step()

        logger.info("Training completed.")
