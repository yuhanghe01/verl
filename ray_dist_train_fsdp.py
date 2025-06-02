import os
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


# ---------- Dummy model and dataset ----------

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(10, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.seq(x)

class RandomDataset(Dataset):
    def __init__(self, size=10000):
        self.x = torch.randn(size, 10)
        self.y = torch.randn(size, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ---------- FSDP Ray Worker ----------

@ray.remote(num_cpus=2, num_gpus=1)
class FSDPWorker:
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

    def train(self, num_epochs=3):
        self.setup()
        device = torch.device("cuda", self.rank % torch.cuda.device_count())

        # Create model and wrap with FSDP
        model = SimpleModel().to(device)
        fsdp_model = FSDP(model, device_id=device)

        optimizer = optim.Adam(fsdp_model.parameters(), lr=0.001)
        dataset = RandomDataset()
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)
        dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            total_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = nn.MSELoss()(fsdp_model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Rank {self.rank}] Epoch {epoch}, Loss: {total_loss:.4f}")

        dist.destroy_process_group()
        return f"Rank {self.rank} done."


# ---------- Launcher ----------

def main():
    ray.init(address="auto")  # Connect to Ray cluster

    world_size = 16  # 2 nodes × 8 GPUs
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "23456")

    print(f"Launching FSDP with {world_size} Ray actors...")

    # Launch FSDP workers
    workers = [
        FSDPWorker.remote(
            rank=i,
            world_size=world_size,
            master_addr=master_addr,
            master_port=master_port,
        )
        for i in range(world_size)
    ]

    futures = [w.train.remote(num_epochs=3) for w in workers]
    results = ray.get(futures)

    for res in results:
        print(res)


if __name__ == "__main__":
    main()

