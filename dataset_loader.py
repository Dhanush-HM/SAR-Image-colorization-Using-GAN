import os
import scipy.io
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Dataset class
class SAR2RGBDataset(Dataset):

    def __init__(self, data_path):

        self.data_path = data_path

        self.files = os.listdir(data_path)

        # Separate SAR and RGB files
        self.mat_files = [f for f in self.files if f.endswith('.mat')]
        self.png_files = [f for f in self.files if f.endswith('.png')]

        # Transform for RGB image
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):

        # SAR file
        mat_file = self.mat_files[idx]

        # Extract point ID
        point_id = mat_file.split('_')[1]

        # Find matching RGB image
        matching_png = None

        for png in self.png_files:
            if point_id in png:
                matching_png = png
                break

        # Load SAR data
        mat_data = scipy.io.loadmat(
            os.path.join(self.data_path, mat_file)
        )

        sar_image = mat_data['ampCrop']
        # Normalize SAR image
        sar_image = (sar_image - sar_image.min()) / (sar_image.max() - sar_image.min())

        # Convert SAR to tensor
        sar_tensor = torch.tensor(sar_image, dtype=torch.float32)

        # Add channel dimension
        sar_tensor = sar_tensor.unsqueeze(0)

        # Normalize to [-1,1]
        sar_tensor = (sar_tensor * 2) - 1

        # Load RGB image
        rgb_image = Image.open(
            os.path.join(self.data_path, matching_png)
        ).convert("RGB")

        rgb_tensor = self.transform_rgb(rgb_image)

        return sar_tensor, rgb_tensor


# Dataset path
data_path = r"datasets/patch_SAR_OPT_SQUARE"

# Create dataset
dataset = SAR2RGBDataset(data_path)

print("Dataset Size:", len(dataset))

# Test sample
sar, rgb = dataset[0]

print("SAR Tensor Shape:", sar.shape)
print("RGB Tensor Shape:", rgb.shape)

print("SAR Min:", sar.min().item())
print("SAR Max:", sar.max().item())

print("RGB Min:", rgb.min().item())
print("RGB Max:", rgb.max().item())