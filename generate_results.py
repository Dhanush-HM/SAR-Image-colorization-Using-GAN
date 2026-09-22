import torch
import glob
import os
import matplotlib.pyplot as plt

from dataset_loader import SAR2RGBDataset
from generator import Generator

# -------------------------
# USER SETTINGS
# -------------------------

# Starting dataset index
start_index = 0

# Number of images to display
num_images = 2

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load Generator
# -------------------------
generator = Generator().to(device)

# -------------------------
# Select Checkpoint
# -------------------------

epoch_number = int(input("Enter checkpoint epoch number: "))

checkpoint_file = f"checkpoint_epoch_{epoch_number}.pth"

if not os.path.exists(checkpoint_file):
    print(f"Checkpoint not found: {checkpoint_file}")
    exit()

print(f"Loading: {checkpoint_file}")

checkpoint = torch.load(
    checkpoint_file,
    map_location=device
)

generator.load_state_dict(
    checkpoint["generator_state_dict"]
)

generator.eval()

print("Generator loaded successfully.")

# -------------------------
# Load Dataset
# -------------------------
data_path = r"datasets/patch_SAR_OPT_SQUARE"

dataset = SAR2RGBDataset(data_path)

# -------------------------
# Plot Results
# -------------------------
plt.figure(figsize=(12, num_images * 4))

for i in range(num_images):

    # Get dataset sample
    sar, real_rgb = dataset[start_index + i]

    # Add batch dimension
    sar_input = sar.unsqueeze(0).to(device)

    # Generate fake RGB
    with torch.no_grad():
        fake_rgb = generator(sar_input)

    # Remove batch dimension
    fake_rgb = fake_rgb.squeeze(0).cpu()

    # Convert tensors
    sar = sar.squeeze(0).numpy()

    real_rgb = real_rgb.permute(1,2,0).numpy()

    fake_rgb = fake_rgb.permute(1,2,0).numpy()

    # Convert [-1,1] → [0,1]
    real_rgb = (real_rgb + 1) / 2

    fake_rgb = (fake_rgb + 1) / 2

    # -------------------------
    # SAR
    # -------------------------
    plt.subplot(num_images, 3, i*3 + 1)

    plt.imshow(sar, cmap='gray')

    plt.title(f"SAR #{start_index+i}")

    plt.axis('off')

    # -------------------------
    # REAL RGB
    # -------------------------
    plt.subplot(num_images, 3, i*3 + 2)

    plt.imshow(real_rgb)

    plt.title("Real RGB")

    plt.axis('off')

    # -------------------------
    # GENERATED RGB
    # -------------------------
    plt.subplot(num_images, 3, i*3 + 3)

    plt.imshow(fake_rgb)

    plt.title("Generated RGB")

    plt.axis('off')

plt.tight_layout()

plt.show()