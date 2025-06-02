import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import ray
from ray.train._internal.worker_group import WorkerGroup
from ray.train._internal.resource_pool import RayResourcePool
from ray.train._internal.worker_group import _RayWorkerGroup
from ray.train._internal.session import _set_internal_session
from ray.air.util.check_ingress import RayClassWithInitArgs

# ----------- Dummy model and dataset ------------

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self, size=10000):
        self.x = torch.randn(size, 10)
        self.y = torch.randn(size, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# ---------- Per-worker class for training -----------

class Worker:
    def __init__(self, rank, world_size, master_addr, master_port):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port

    def setup(self):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(self.rank % torch.cuda.device_count())
        print(f"[Rank {self.rank}] initialized on GPU {torch.cuda.current_device()}.")

    def train(self):
        self.setup()
        rank = self.rank
        device = torch.device("cuda", rank % torch.cuda.device_count())

        model = SimpleModel().to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        dataset = RandomDataset()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=self.world_size, rank=rank)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=sampler)

        for epoch in range(3):
            total_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Rank {rank}] Epoch {epoch}, Loss: {total_loss:.4f}")

        dist.destroy_process_group()

# ------------- Ray training orchestration --------------

def main():
    ray.init(address="auto")  # connect to existing Ray cluster

    world_size = 16  # 8 GPUs x 2 nodes
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "23456")

    print(f"Launching {world_size} training workers...")

    ray_worker_cls = RayClassWithInitArgs(
        Worker,
        init_kwargs={
            "rank": 0,  # placeholder, will override below
            "world_size": world_size,
            "master_addr": master_addr,
            "master_port": master_port
        }
    )

    # Create 16 workers (each needs 1 GPU)
    pool = RayResourcePool(
        ray_worker_cls,
        num_workers=world_size,
        resources_per_worker={"CPU": 2, "GPU": 1},
        max_concurrent_tasks=1,
    )
    pool.start()

    # Assign rank manually to each actor
    workers = pool.get_workers()
    for rank, w in enumerate(workers):
        w.__ray_actor__.update_init_args.remote(
            init_kwargs={
                "rank": rank,
                "world_size": world_size,
                "master_addr": master_addr,
                "master_port": master_port,
            }
        )

    worker_group = WorkerGroup(workers)
    worker_group.execute(lambda w: w.train())

    print("Distributed training completed.")
    pool.shutdown()

if __name__ == "__main__":
    main()
