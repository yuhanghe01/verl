"""
SFT_training.py
===============
Supervised fine-tuning (SFT) pipeline for Azure HPC InfiniBand incident
assistants, with distributed (``torchrun``) training *and* evaluation.

Dataset
-------
JSONL chat records (one JSON object per line) with the schema::

    {
      "task_type": "classification" | "summarization",
      "messages": [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "<target>"}   # supervision target
      ],
      ...
    }

  * ``classification``      -> the assistant message is a single family label.
  * ``summarization`` (etc.) -> the assistant message is free-form text.

Base model
----------
Customizable via ``--model_name_or_path``.  Tested intent:
  * ``Qwen/Qwen2.5-32B-Instruct``        (default)
  * ``Qwen/Qwen2.5-Coder-32B-Instruct``  (the "Qwen-32B-Coder" variant)

Evaluation metrics
------------------
  * classification     -> Accuracy (exact label match, normalized).
  * non-classification -> LLM-as-Judge similarity score in [0, 1].

Distributed usage (torchrun)
----------------------------
Train (LoRA, fits 32B on 8x80GB; drop ``--use_lora`` + add ``--fsdp`` for full FT)::

    torchrun --nproc_per_node=8 SFT_training.py \
        --do_train --do_eval \
        --model_name_or_path Qwen/Qwen2.5-32B-Instruct \
        --train_file stage1_full_85_15/stage1_train.jsonl \
        --dev_file   stage1_full_85_15/stage1_dev.jsonl \
        --output_dir ./sft_qwen32b \
        --use_lora \
        --bf16 True --gradient_checkpointing True \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
        --learning_rate 1e-4 --warmup_ratio 0.03 --logging_steps 10 \
        --save_strategy epoch --max_seq_length 4096

Full fine-tuning with FSDP instead of LoRA, append e.g.::

    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap Qwen2DecoderLayer

Evaluate only (e.g. a saved checkpoint)::

    torchrun --nproc_per_node=8 SFT_training.py \
        --do_eval \
        --model_name_or_path ./sft_qwen32b \
        --dev_file stage1_full_85_15/stage1_dev.jsonl \
        --output_dir ./sft_qwen32b
"""

from __future__ import annotations

import json
import os
import re
import math
import inspect
import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:  # pragma: no cover - FSDP unavailable on some builds
    FSDP = None

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

IGNORE_INDEX = -100
CLASSIFICATION_TASKS = {"classification"}


# ---------------------------------------------------------------------------
# Data IO
# ---------------------------------------------------------------------------
def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_records(paths: Iterable[str | Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return records


# ---------------------------------------------------------------------------
# Argument definitions
# ---------------------------------------------------------------------------
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-32B-Instruct",
        metadata={"help": "HF id or local path. e.g. Qwen/Qwen2.5-32B-Instruct "
                          "or Qwen/Qwen2.5-Coder-32B-Instruct."},
    )
    trust_remote_code: bool = field(default=True)
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "e.g. flash_attention_2 / sdpa / eager."},
    )
    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA (PEFT)."})
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated module names for LoRA."},
    )


@dataclass
class DataArguments:
    train_file: Optional[str] = field(default="stage1_full_85_15/stage1_train.jsonl")
    dev_file: Optional[str] = field(default="stage1_full_85_15/stage1_dev.jsonl")
    max_seq_length: int = field(default=4096)


@dataclass
class EvalArguments:
    eval_batch_size: int = field(default=4)
    eval_max_new_tokens: int = field(default=512)
    class_max_new_tokens: int = field(
        default=32, metadata={"help": "Shorter generation budget for labels."})
    judge_model: Optional[str] = field(
        default=None,
        metadata={"help": "Judge model id/path. Defaults to the policy model "
                          "itself (self-judge) to avoid loading a 2nd 32B model."},
    )
    judge_max_new_tokens: int = field(default=16)
    judge_api_base: Optional[str] = field(
        default=None,
        metadata={"help": "OpenAI-compatible endpoint for the judge. If set, "
                          "uses it instead of a local judge model."},
    )
    judge_api_key: Optional[str] = field(default=None)
    judge_api_model: Optional[str] = field(default="gpt-4o-mini")


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def maybe_init_distributed() -> torch.device:
    """Initialise the default process group from torchrun env vars (idempotent)."""
    if dist.is_available() and not dist.is_initialized() and "RANK" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def all_gather_objects(obj: Any) -> List[Any]:
    if not is_dist():
        return [obj]
    gathered: List[Any] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def rprint(*args: Any) -> None:
    if is_main():
        print(*args, flush=True)


# ---------------------------------------------------------------------------
# Tokenizer / model
# ---------------------------------------------------------------------------
def build_tokenizer(model_args: ModelArguments) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def build_model(model_args: ModelArguments, training_args: TrainingArguments):
    dtype = (torch.bfloat16 if training_args.bf16
             else (torch.float16 if training_args.fp16 else None))
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=dtype,
        attn_implementation=model_args.attn_implementation,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    if getattr(training_args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model
        lconf = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[m.strip() for m in model_args.lora_target_modules.split(",") if m.strip()],
        )
        model = get_peft_model(model, lconf)
        if is_main():
            model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Dataset / collator
# ---------------------------------------------------------------------------
class SFTDataset(Dataset):
    """Tokenizes chat records, masking the prompt so loss is on the target only."""

    def __init__(self, records: List[Dict[str, Any]],
                 tokenizer: PreTrainedTokenizerBase, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        # Keep only records with a final assistant target.
        self.records = [r for r in records
                        if r.get("messages") and r["messages"][-1]["role"] == "assistant"]

    def __len__(self) -> int:
        return len(self.records)

    def _apply_template_ids(self, messages: List[Dict[str, str]],
                            add_generation_prompt: bool) -> List[int]:
        """Return a flat ``list[int]`` of token ids from the chat template.

        Depending on the tokenizer / transformers version, ``apply_chat_template``
        may return a plain list, a nested ``[[...]]`` list, or a ``BatchEncoding``
        dict. Normalise all of these to a flat list of ints.
        """
        out = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )
        # BatchEncoding / dict -> take the input_ids field.
        if isinstance(out, dict):
            out = out["input_ids"]
        # Batched output [[...]] -> unwrap the single example.
        if len(out) > 0 and isinstance(out[0], (list, tuple)):
            out = out[0]
        return list(out)

    def _encode(self, messages: List[Dict[str, str]]) -> Tuple[List[int], List[int]]:
        prompt_ids = self._apply_template_ids(messages[:-1], add_generation_prompt=True)
        full_ids = self._apply_template_ids(messages, add_generation_prompt=False)
        resp_ids = full_ids[len(prompt_ids):]

        max_len = self.max_seq_length
        if len(prompt_ids) + len(resp_ids) > max_len:
            if len(resp_ids) >= max_len:
                # Pathologically long target: keep its head, no prompt.
                resp_ids = resp_ids[:max_len]
                prompt_ids = []
            else:
                # Keep the *tail* of the prompt: preserves the generation-prompt
                # suffix (assistant header) and the most recent context.
                prompt_ids = prompt_ids[-(max_len - len(resp_ids)):]

        input_ids = prompt_ids + resp_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + list(resp_ids)
        return input_ids, labels

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        input_ids, labels = self._encode(self.records[idx]["messages"])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


@dataclass
class DataCollatorForSFT:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            n = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [pad_id] * n)
            labels.append(f["labels"] + [IGNORE_INDEX] * n)
            attn.append(f["attention_mask"] + [0] * n)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_training(model, tokenizer, data_args: DataArguments,
                 training_args: TrainingArguments,
                 eval_args: Optional[EvalArguments] = None) -> None:
    """Explicit training loop (no HF ``Trainer``) so intermediate results such
    as per-step loss / learning-rate / grad-norm can be printed directly."""
    device = maybe_init_distributed()
    train_records = load_records([data_args.train_file])
    train_dataset = SFTDataset(train_records, tokenizer, data_args.max_seq_length)
    rprint(f"[train] {len(train_dataset)} examples")

    # Choose the parallelism wrapper:
    #   * FSDP (full/grad-op shard) when --fsdp is passed -> needed for full
    #     fine-tuning of large models (shards params/grads/optimizer states).
    #   * DDP otherwise (replicates everything; fine for LoRA / small models).
    use_fsdp = (is_dist() and torch.cuda.is_available()
                and FSDP is not None
                and bool(getattr(training_args, "fsdp", None)))
    if use_fsdp:
        # FSDP shards from CPU via ``device_id``; do NOT pre-move the full model
        # to GPU (that would need the whole 32B on a single device).
        model = _wrap_fsdp(model, training_args, device)
    elif is_dist() and torch.cuda.is_available():
        model.to(device)
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
    else:
        model.to(device)
    base_model = model.module if hasattr(model, "module") else model

    # ---- DataLoader -------------------------------------------------------
    sampler = (DistributedSampler(train_dataset, shuffle=True, seed=training_args.seed)
               if is_dist() else None)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=DataCollatorForSFT(tokenizer),
        num_workers=getattr(training_args, "dataloader_num_workers", 0),
        pin_memory=True,
        drop_last=False,
    )

    grad_accum = max(1, training_args.gradient_accumulation_steps)
    num_epochs = max(1, int(training_args.num_train_epochs))
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    max_steps = steps_per_epoch * num_epochs

    # ---- Optimizer / scheduler -------------------------------------------
    # Iterate over the (possibly FSDP-wrapped) model so the optimizer tracks the
    # sharded ``use_orig_params`` tensors under FSDP, and the replicated tensors
    # under DDP / single-GPU.
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": training_args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    scheduler = transformers.get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.get_warmup_steps(max_steps),
        num_training_steps=max_steps,
    )

    rprint(f"[train] epochs={num_epochs} steps/epoch={steps_per_epoch} "
           f"total_optim_steps={max_steps} grad_accum={grad_accum} "
           f"world_size={get_world_size()}")

    # ---- Training loop ----------------------------------------------------
    log_every = int(training_args.logging_steps) if training_args.logging_steps else 0
    global_step = 0
    model.train()
    for epoch in range(num_epochs):
        rprint(f"[train] starting epoch {epoch + 1}/{num_epochs}")
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        micro_loss_sum, micro_count = 0.0, 0
        n_batches = len(train_loader)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum

            at_boundary = ((step + 1) % grad_accum == 0) or (step + 1 == n_batches)
            # Avoid all-reduce on non-boundary micro-steps for DDP efficiency.
            if isinstance(model, DDP) and not at_boundary:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()

            micro_loss_sum += loss.item() * grad_accum
            micro_count += 1

            if at_boundary:
                if training_args.max_grad_norm and training_args.max_grad_norm > 0:
                    if use_fsdp:
                        # FSDP grads are sharded; use its collective clip helper.
                        grad_norm = model.clip_grad_norm_(training_args.max_grad_norm)
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            training_args.max_grad_norm,
                        )
                else:
                    grad_norm = None
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if log_every and global_step % log_every == 0:
                    avg_loss = micro_loss_sum / max(1, micro_count)
                    micro_loss_sum, micro_count = 0.0, 0
                    lr = scheduler.get_last_lr()[0]
                    gn = f"{float(grad_norm):.3f}" if grad_norm is not None else "n/a"
                    rprint(f"[train] epoch {epoch + 1}/{num_epochs} "
                           f"step {global_step}/{max_steps} "
                           f"loss {avg_loss:.4f} lr {lr:.3e} grad_norm {gn}")

        rprint(f"[train] finished epoch {epoch + 1}/{num_epochs}")

        # Generation-based evaluation at the end of every epoch.
        if training_args.do_eval and eval_args is not None:
            # FSDP needs the wrapped root for ``summon_full_params`` during
            # generation; DDP / single-GPU can generate on the unwrapped model.
            gen_model = model if use_fsdp else base_model
            _run_epoch_eval(gen_model, base_model, tokenizer, data_args,
                            eval_args, training_args, device, epoch + 1)
            model.train()

        # Save a separate checkpoint for this epoch under epoch_xx/.
        _save_model(model, base_model, tokenizer, training_args, use_fsdp,
                    epoch=epoch + 1)
        if is_dist():
            dist.barrier()


def _run_epoch_eval(gen_model, base_model, tokenizer, data_args: DataArguments,
                    eval_args: EvalArguments, training_args: TrainingArguments,
                    device: torch.device, epoch: int) -> None:
    """Toggle inference-friendly settings and run generation-based evaluation.

    ``gen_model`` is the object passed to generation (the FSDP root when
    sharding, so its parameters can be summoned); ``base_model`` is the
    underlying HF model used to flip ``use_cache`` / gradient checkpointing.
    """
    was_training = base_model.training
    cfg = getattr(base_model, "config", None)
    prev_use_cache = getattr(cfg, "use_cache", None) if cfg is not None else None
    gc_enabled = getattr(base_model, "is_gradient_checkpointing", False)
    try:
        if gc_enabled:
            base_model.gradient_checkpointing_disable()
        if cfg is not None:
            cfg.use_cache = True
        gen_model.eval()
        run_evaluation(gen_model, tokenizer, data_args, eval_args,
                       training_args, device, tag=f"epoch{epoch}")
    finally:
        if cfg is not None and prev_use_cache is not None:
            cfg.use_cache = prev_use_cache
        if gc_enabled:
            base_model.gradient_checkpointing_enable()
        if was_training:
            gen_model.train()


def _wrap_fsdp(model, training_args: TrainingArguments, device: torch.device):
    """Wrap ``model`` in FullyShardedDataParallel for sharded full fine-tuning.

    Sharding strategy is taken from the ``--fsdp`` options and the transformer
    layer class to auto-wrap from ``--fsdp_config``'s
    ``transformer_layer_cls_to_wrap`` key.
    """
    import functools
    from torch.distributed.fsdp import (
        MixedPrecision,
        ShardingStrategy,
        BackwardPrefetch,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    options = {str(o).lower() for o in (training_args.fsdp or [])}
    if "shard_grad_op" in options or "_shard_grad_op" in options:
        strategy = ShardingStrategy.SHARD_GRAD_OP
    elif "hybrid_shard" in options:
        strategy = ShardingStrategy.HYBRID_SHARD
    elif "no_shard" in options:
        strategy = ShardingStrategy.NO_SHARD
    else:
        strategy = ShardingStrategy.FULL_SHARD

    fsdp_cfg = getattr(training_args, "fsdp_config", None) or {}
    wrap_names = set(fsdp_cfg.get("transformer_layer_cls_to_wrap", []) or [])
    layer_classes = {m.__class__ for m in model.modules()
                     if m.__class__.__name__ in wrap_names}
    auto_wrap_policy = (
        functools.partial(transformer_auto_wrap_policy,
                          transformer_layer_cls=layer_classes)
        if layer_classes else None)
    if not layer_classes and is_main():
        rprint(f"[train][fsdp] warning: none of {sorted(wrap_names)} matched a "
               f"module class; falling back to size-based auto-wrap.")

    mp_dtype = (torch.bfloat16 if training_args.bf16
                else (torch.float16 if training_args.fp16 else None))
    mixed_precision = (MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype,
                                      buffer_dtype=mp_dtype)
                       if mp_dtype is not None else None)

    rprint(f"[train][fsdp] strategy={strategy.name} "
           f"wrap={sorted(c.__name__ for c in layer_classes) or 'size-based'} "
           f"mixed_precision={mp_dtype}")
    return FSDP(
        model,
        sharding_strategy=strategy,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        use_orig_params=True,
        limit_all_gathers=True,
    )


def _save_model(model, base_model, tokenizer, training_args: TrainingArguments,
                use_fsdp: bool, epoch: Optional[int] = None) -> None:
    """Save the (possibly sharded) model + tokenizer.

    When ``epoch`` is given the checkpoint is written to an ``epoch_xx``
    subfolder of ``output_dir``; otherwise it is written to ``output_dir``.
    """
    out_dir = Path(training_args.output_dir)
    if epoch is not None:
        out_dir = out_dir / f"epoch_{epoch:02d}"
    if use_fsdp:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # Gather the full (unsharded) state dict onto rank 0 / CPU.
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
            cpu_state = model.state_dict()
        if is_main():
            out_dir.mkdir(parents=True, exist_ok=True)
            base_model.save_pretrained(str(out_dir), state_dict=cpu_state)
            tokenizer.save_pretrained(str(out_dir))
            rprint(f"[train] saved model to {out_dir}")
    else:
        if is_dist():
            dist.barrier()
        if is_main():
            out_dir.mkdir(parents=True, exist_ok=True)
            base_model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            rprint(f"[train] saved model to {out_dir}")


def _detect_resume(training_args: TrainingArguments) -> Optional[bool]:
    try:
        ckpts = list(Path(training_args.output_dir).glob("checkpoint-*"))
        return True if ckpts else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Evaluation: distributed generation + metrics
# ---------------------------------------------------------------------------
def _label_set_from_prompt(user_content: str) -> List[str]:
    """Parse 'Allowed family labels: a, b, c' from a classification prompt."""
    m = re.search(r"Allowed[^:]*labels?:\s*(.+)", user_content, flags=re.IGNORECASE)
    if not m:
        return []
    line = m.group(1).splitlines()[0]
    return [t.strip() for t in re.split(r"[,\n]", line) if t.strip()]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _match_label(prediction: str, labels: List[str]) -> str:
    """Resolve a free-form generation to one of the allowed labels."""
    pred = _normalize(prediction)
    norm = {_normalize(l): l for l in labels}
    if pred in norm:
        return norm[pred]
    # First allowed label that appears as a token/substring in the prediction.
    for nl, orig in sorted(norm.items(), key=lambda kv: -len(kv[0])):
        if re.search(r"\b" + re.escape(nl) + r"\b", pred):
            return orig
    for nl, orig in sorted(norm.items(), key=lambda kv: -len(kv[0])):
        if nl in pred:
            return orig
    return prediction.strip()


@torch.no_grad()
def _generate_batch(model, tokenizer, prompts: List[str], device: torch.device,
                    max_new_tokens: int) -> List[str]:
    tokenizer.padding_side = "left"
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    enc = {k: v.to(device) for k, v in enc.items()}

    # Under FSDP the parameters are sharded (the embedding weight is a 1-D flat
    # shard), so calling generate() directly raises "'weight' must be 2-D".
    # Gather the full parameters for the duration of generation.
    use_fsdp = (FSDP is not None
                and any(isinstance(m, FSDP) for m in model.modules()))
    if use_fsdp:
        gather_ctx = FSDP.summon_full_params(model, recurse=True, writeback=False)
    else:
        gather_ctx = contextlib.nullcontext()

    with gather_ctx:
        base = model.module if hasattr(model, "module") else model
        out = base.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    gen = out[:, enc["input_ids"].shape[1]:]
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def _render_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True)


def run_generation(model, tokenizer, records: List[Dict[str, Any]],
                   device: torch.device, batch_size: int,
                   max_new_tokens: int) -> List[Dict[str, Any]]:
    """Shard `records` across ranks, generate, return per-record predictions."""
    rank, world = get_rank(), get_world_size()
    shard = records[rank::world]
    results: List[Dict[str, Any]] = []
    for i in range(0, len(shard), batch_size):
        batch = shard[i:i + batch_size]
        prompts = [_render_prompt(tokenizer, r["messages"]) for r in batch]
        preds = _generate_batch(model, tokenizer, prompts, device, max_new_tokens)
        for r, p in zip(batch, preds):
            results.append({
                "task_type": r.get("task_type", "summarization"),
                "user": r["messages"][-2]["content"],
                "reference": r["messages"][-1]["content"],
                "prediction": p,
            })
    gathered = all_gather_objects(results)
    flat: List[Dict[str, Any]] = []
    for part in gathered:
        flat.extend(part)
    return flat


# ---- LLM-as-Judge -----------------------------------------------------------
JUDGE_SYSTEM = (
    "You are a strict evaluation judge. Compare a MODEL ANSWER to a REFERENCE "
    "answer and rate how well the model answer conveys the same factual content "
    "and meaning as the reference. Respond with ONLY a single integer from 1 to "
    "10, where 10 means semantically equivalent and 1 means unrelated."
)


def _judge_prompt_messages(user: str, reference: str, prediction: str) -> List[Dict[str, str]]:
    content = (
        f"TASK CONTEXT (input given to the model):\n{user}\n\n"
        f"REFERENCE ANSWER:\n{reference}\n\n"
        f"MODEL ANSWER:\n{prediction}\n\n"
        "Score (1-10):"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": content}]


def _parse_score(text: str) -> Optional[float]:
    m = re.search(r"(10|[1-9])(?:\.0)?", text)
    if not m:
        return None
    return max(0.0, min(1.0, float(m.group(1)) / 10.0))


def judge_local(model, tokenizer, pairs: List[Dict[str, Any]], device: torch.device,
                eval_args: EvalArguments) -> List[float]:
    """Self/local-model judge. Shards across ranks like generation."""
    rank, world = get_rank(), get_world_size()
    idx = list(range(len(pairs)))
    shard_idx = idx[rank::world]
    local: List[Tuple[int, float]] = []
    bs = eval_args.eval_batch_size
    for i in range(0, len(shard_idx), bs):
        chunk = shard_idx[i:i + bs]
        prompts = [
            tokenizer.apply_chat_template(
                _judge_prompt_messages(pairs[j]["user"], pairs[j]["reference"],
                                       pairs[j]["prediction"]),
                tokenize=False, add_generation_prompt=True)
            for j in chunk
        ]
        outs = _generate_batch(model, tokenizer, prompts, device,
                               eval_args.judge_max_new_tokens)
        for j, o in zip(chunk, outs):
            s = _parse_score(o)
            local.append((j, s if s is not None else 0.0))
    gathered = all_gather_objects(local)
    scores = [0.0] * len(pairs)
    for part in gathered:
        for j, s in part:
            scores[j] = s
    return scores


def judge_api(pairs: List[Dict[str, Any]], eval_args: EvalArguments) -> List[float]:
    """OpenAI-compatible judge (rank 0 only)."""
    from openai import OpenAI
    client = OpenAI(base_url=eval_args.judge_api_base,
                    api_key=eval_args.judge_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
    scores: List[float] = []
    for p in pairs:
        resp = client.chat.completions.create(
            model=eval_args.judge_api_model,
            messages=_judge_prompt_messages(p["user"], p["reference"], p["prediction"]),
            temperature=0.0, max_tokens=eval_args.judge_max_new_tokens)
        s = _parse_score(resp.choices[0].message.content or "")
        scores.append(s if s is not None else 0.0)
    return scores


def run_evaluation(model, tokenizer, data_args: DataArguments,
                   eval_args: EvalArguments, training_args: TrainingArguments,
                   device: torch.device, tag: Optional[str] = None) -> Dict[str, Any]:
    records = load_records([data_args.dev_file])
    rprint(f"[eval] {len(records)} dev examples on {get_world_size()} rank(s)")
    model.eval()

    # 1) Generate predictions for every dev record (sharded across ranks).
    class_records = [r for r in records if r.get("task_type") in CLASSIFICATION_TASKS]
    gen_records = [r for r in records if r.get("task_type") not in CLASSIFICATION_TASKS]

    class_preds = run_generation(model, tokenizer, class_records, device,
                                 eval_args.eval_batch_size, eval_args.class_max_new_tokens)
    gen_preds = run_generation(model, tokenizer, gen_records, device,
                               eval_args.eval_batch_size, eval_args.eval_max_new_tokens)

    # 2a) Classification -> Accuracy.
    correct = 0
    for item in class_preds:
        labels = _label_set_from_prompt(item["user"])
        pred_label = _match_label(item["prediction"], labels)
        item["pred_label"] = pred_label
        if _normalize(pred_label) == _normalize(item["reference"]):
            correct += 1
    accuracy = (correct / len(class_preds)) if class_preds else float("nan")

    # 2b) Non-classification -> LLM-as-Judge similarity.
    # API-based judge is forcefully disabled; always use the local self-judge.
    if gen_preds:
        judge_scores = judge_local(model, tokenizer, gen_preds, device, eval_args)
        for item, s in zip(gen_preds, judge_scores):
            item["judge_score"] = s
        judge_mean = sum(judge_scores) / len(judge_scores)
    else:
        judge_mean = float("nan")

    metrics = {
        "num_classification": len(class_preds),
        "classification_accuracy": accuracy,
        "num_generation": len(gen_preds),
        "judge_similarity_mean": judge_mean,
    }
    if tag:
        metrics["tag"] = tag
    if is_main():
        out_dir = Path(training_args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        with (out_dir / f"eval_metrics{suffix}.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        with (out_dir / f"eval_predictions{suffix}.jsonl").open("w", encoding="utf-8") as f:
            for item in class_preds + gen_preds:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"==== Evaluation metrics{(' [' + tag + ']') if tag else ''} ====")
        print(json.dumps(metrics, indent=2))
    return metrics


class PerEpochEvalCallback(TrainerCallback):
    """Run the generation-based evaluation at the end of every training epoch.

    Results are written to ``eval_metrics_epoch{N}.json`` /
    ``eval_predictions_epoch{N}.jsonl`` under ``output_dir`` so per-epoch runs
    do not overwrite one another.
    """

    def __init__(self, tokenizer, data_args: DataArguments,
                 eval_args: EvalArguments, device: torch.device):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.eval_args = eval_args
        self.device = device
        self.trainer = None  # set by run_training; gives access to model_wrapped

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        # Prefer the (possibly FSDP/DDP) wrapped model so parameters can be
        # gathered during generation; fall back to the model passed in.
        eval_model = model
        if self.trainer is not None and getattr(self.trainer, "model_wrapped", None) is not None:
            eval_model = self.trainer.model_wrapped
        if eval_model is None:
            return control
        epoch = int(round(state.epoch)) if state.epoch else state.global_step
        was_training = eval_model.training
        cfg = getattr(eval_model, "config", None)
        prev_use_cache = getattr(cfg, "use_cache", None) if cfg is not None else None
        gc_enabled = getattr(eval_model, "is_gradient_checkpointing", False)
        try:
            if gc_enabled:
                eval_model.gradient_checkpointing_disable()
            if cfg is not None:
                cfg.use_cache = True
            eval_model.eval()
            run_evaluation(eval_model, self.tokenizer, self.data_args,
                           self.eval_args, args, self.device, tag=f"epoch{epoch}")
        finally:
            if cfg is not None and prev_use_cache is not None:
                cfg.use_cache = prev_use_cache
            if gc_enabled:
                eval_model.gradient_checkpointing_enable()
            if was_training:
                eval_model.train()
        return control


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, EvalArguments, TrainingArguments))
    model_args, data_args, eval_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)

    tokenizer = build_tokenizer(model_args)

    if training_args.do_train:
        model = build_model(model_args, training_args)
        print('starting the training')
        run_training(model, tokenizer, data_args, training_args, eval_args)

    # During training the PerEpochEvalCallback already evaluates after every
    # epoch, so a separate evaluation is only needed when running eval-only.
    if training_args.do_eval and not training_args.do_train:
        device = maybe_init_distributed()
        # Standalone evaluation: load the (possibly fine-tuned) model directly.
        dtype = (torch.bfloat16 if training_args.bf16
                 else (torch.float16 if training_args.fp16 else None))
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=dtype,
            attn_implementation=model_args.attn_implementation,
        )
        model.to(device)
        model.config.use_cache = True
        run_evaluation(model, tokenizer, data_args, eval_args, training_args, device)

    if is_dist():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
