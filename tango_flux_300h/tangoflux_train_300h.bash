#!/bin/bash
# Train TangoFlux with accelerate on GPUs 0 and 1

# Select GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Launch training FT
accelerate launch \
  --config_file="accelerator_config.yaml" \
  /opt/conda/envs/tango/lib/python3.10/site-packages/tangoflux/train.py \
  --checkpointing_steps=best \
  --save_every=1 \
  --config="tangoflux_config_ft.yaml" \
  --load_from_checkpoint /mnt/blob-data-sigmasystem/yuhang/hub/models--declare-lab--TangoFlux/snapshots/367005e963cb3a9fb2e03a46104d7de23e34ceea/tangoflux.safetensors \


accelerate launch \
  --config_file="accelerator_config.yaml" \
  /opt/conda/envs/tango/lib/python3.10/site-packages/tangoflux/train.py \
  --checkpointing_steps=best \
  --save_every=5 \
  --config="tangoflux_config_scratch.yaml"
