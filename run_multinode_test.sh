torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --node_rank=$PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX \
  --rdzv_backend=c10d \
  --rdzv_endpoint=worker-0:29500 \
  multinode_ddp_test.py
