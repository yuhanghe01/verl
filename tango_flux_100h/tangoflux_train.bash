#!/bin/bash
# Train TangoFlux with accelerate on GPUs 0 and 1

# Select GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


ckpt_dir='/mnt/blob-data-sigmasystem-out/yuhang/Aurelius/tangoflux_100h_finetune'
mkdir $ckpt_dir

# Launch training
accelerate launch \
  --config_file="accelerator_config.yaml" \
  train.py \
  --checkpointing_steps=best \
  --save_every=5 \
  --config="tangoflux_config.yaml" \
  --load_from_checkpoint /root/.cache/huggingface/hub/models--declare-lab--TangoFlux/snapshots/367005e963cb3a9fb2e03a46104d7de23e34ceea/tangoflux.safetensors \
