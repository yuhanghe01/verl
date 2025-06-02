import os
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

# ---------- Dummy model and dataset ----------

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

# ---------- Training Actor Class ----------

@ray.remote(num_cpus=2, num_gpus=1)
class TrainerWorker:
    def __init__(self, rank, world_size, master_addr, master_port):
        self.rank = rank
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port

    def setup_distributed(self):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        dist.init_process_group("nccl")
        torch.cuda.set_device(self.rank % torch.cuda.device_count())

    def train(self, num_epochs=3):
        self.setup_distributed()
        device = torch.device("cuda", self.rank % torch.cuda.device_count())

        model = SimpleModel().to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device])
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()

        dataset = RandomDataset()
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self.rank)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=sampler)

        for epoch in range(num_epochs):
            total_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"[Rank {self.rank}] Epoch {epoch}, Loss: {total_loss:.4f}")

        dist.destroy_process_group()
        return f"Rank {self.rank} finished training."

# ---------- Main Launcher ----------

def main():
    ray.init(address="auto")  # connect to existing Ray cluster

    world_size = 16
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "23456")

    print(f"Launching {world_size} Ray actor workers...")

    # Create remote actors
    workers = [
        TrainerWorker.remote(rank=i, world_size=world_size,
                             master_addr=master_addr, master_port=master_port)
        for i in range(world_size)
    ]

    # Trigger training in parallel
    futures = [w.train.remote(num_epochs=3) for w in workers]
    results = ray.get(futures)

    for r in results:
        print(r)

if __name__ == "__main__":
    main()

