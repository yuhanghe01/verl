import torch
import torch.nn as nn
import torch.optim as optim

# Check available GPUs
if not torch.cuda.is_available():
    print("No AMD GPU found!")
    exit()

num_gpus = torch.cuda.device_count()
print(f"Number of available AMD GPUs: {num_gpus}")

# Dummy model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize model and move to multiple GPUs
model = SimpleModel()
if num_gpus > 1:
    model = nn.DataParallel(model)
model = model.to("cuda")

# Create dummy data
x = torch.randn(16, 10).to("cuda")
y = torch.randn(16, 1).to("cuda")

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Run a simple training step
model.train()
optimizer.zero_grad()
outputs = model(x)
loss = criterion(outputs, y)
loss.backward()
optimizer.step()

print("Multi-GPU test passed. Training step completed successfully!")
