import torch
from torch.utils.data import DataLoader

# Import dataset class
from dataset_loader import SAR2RGBDataset

# Dataset path
data_path = r"datasets/patch_SAR_OPT_SQUARE"

# Create dataset
dataset = SAR2RGBDataset(data_path)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

print("Total Batches:", len(dataloader))

# Get one batch
sar_batch, rgb_batch = next(iter(dataloader))

print("\nSAR Batch Shape:", sar_batch.shape)
print("RGB Batch Shape:", rgb_batch.shape)

print("\nSAR Min:", sar_batch.min().item())
print("SAR Max:", sar_batch.max().item())

print("\nRGB Min:", rgb_batch.min().item())
print("RGB Max:", rgb_batch.max().item())