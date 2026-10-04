"""Synthetic Visual Dataset and Augmentation Pipeline."""
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticImageDataset(Dataset):
    def __init__(self, n_samples: int = 400, img_size: int = 32, n_classes: int = 4, seed: int = 42):
        torch.manual_seed(seed)
        self.n_samples = n_samples
        self.n_classes = n_classes
        # Synthesize distinctive class patterns
        self.images = torch.randn(n_samples, 3, img_size, img_size)
        self.labels = torch.randint(0, n_classes, (n_samples,))
        
        # Inject pattern correlated with class
        for c in range(n_classes):
            mask = (self.labels == c)
            self.images[mask, c % 3, :, :] += 1.5

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Augment with random horizontal flip
        img = self.images[idx]
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, dims=[2])
        return img, self.labels[idx]

def get_dataloaders(batch_size: int = 32):
    train_ds = SyntheticImageDataset(n_samples=400, seed=42)
    test_ds = SyntheticImageDataset(n_samples=100, seed=99)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
