"""PyTorch Residual Convolutional Image Classifier."""
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders

class ConvNet(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16x16
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16)
        )
        self.head = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(32, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = out + self.res_block(out)
        out = self.head(out)
        return out

def train_and_eval(epochs: int = 5):
    train_loader, test_loader = get_dataloaders(batch_size=32)
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            preds = model(imgs).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            
    acc = correct / total
    return model, acc

if __name__ == "__main__":
    model, acc = train_and_eval(epochs=5)
    print(f"Trained Image Classifier Test Accuracy: {acc * 100:.2f}%")
