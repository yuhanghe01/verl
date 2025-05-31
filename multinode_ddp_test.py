import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def setup():
    dist.init_process_group(backend='nccl')

def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.net(x)

def main():
    setup()
    
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    
    model = ToyModel().cuda(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    
    # Dummy dataset
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(5):
        sampler.set_epoch(epoch)
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.cuda(local_rank)
            batch_y = batch_y.cuda(local_rank)
            optimizer.zero_grad()
            outputs = ddp_model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Rank {rank} Epoch {epoch} Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
    main()
