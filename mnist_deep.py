import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
from gpu_helpers import print_details
import deepspeed

print_details()

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Dataset and Dataloader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # mean and std for MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)

torch_dtype=torch.float16

# 2. Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleNN().to(device)



# 3. Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

class Args:
    def __init__(self):
        self.deepspeed_config="ds_config.json"

model_engine, optimizer, _, _ = deepspeed.initialize(args=Args(),model=model, model_parameters=model.parameters(), optimizer=optimizer)

# 4. Training loop
for epoch in range(1, 6):  # 5 epochs
    model.train()
    total_loss = 0
    start=time.time()
    for x, y in train_loader:
        x, y = x.to(device,torch_dtype), y.to(device,torch_dtype)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        model_engine.step(loss)
        total_loss += loss.item()
    end=time.time()
    print(f"Epoch {epoch}, Loss: {total_loss:.2f} elpased {end-start}")

# 5. Evaluation
model.eval()
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()

accuracy = correct / len(test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")
