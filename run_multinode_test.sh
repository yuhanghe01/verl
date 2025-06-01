export NODE_RANK=$PAI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX
export MASTER_ADDR=$PAI_TASK_ROLE_worker_TASK_0_HOST_IP
export MASTER_PORT=29500
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} multinode_ddp_test.py
