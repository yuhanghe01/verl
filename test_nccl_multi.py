import os
import torch
import torch.distributed as dist

def setup():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, local_rank

def cleanup():
    dist.destroy_process_group()

def main():
    rank, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    tensor = torch.ones(1, device=device) * rank
    print(f"[Before AllReduce] Rank {rank}: {tensor}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[After AllReduce] Rank {rank}: {tensor}")

    cleanup()

if __name__ == "__main__":
    main()
