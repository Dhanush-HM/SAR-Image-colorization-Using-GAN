import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Dataset path
data_path = r"datasets/patch_SAR_OPT_SQUARE"

# Get all files
files = os.listdir(data_path)

# Separate SAR and RGB files
mat_files = [f for f in files if f.endswith('.mat')]
png_files = [f for f in files if f.endswith('.png')]

print("Total SAR files:", len(mat_files))
print("Total RGB files:", len(png_files))

# Pick one SAR file
sample_mat = mat_files[0]

# Extract point ID
point_id = sample_mat.split('_')[1]

# Find matching RGB image
matching_png = None

for png in png_files:
    if point_id in png:
        matching_png = png
        break

print("\nSAR File:", sample_mat)
print("RGB File:", matching_png)

# Load SAR data
mat_data = scipy.io.loadmat(os.path.join(data_path, sample_mat))

# Extract SAR image matrix
sar_image = mat_data['ampCrop']

print("\nSAR Shape:", sar_image.shape)

# Load RGB image
rgb_image = Image.open(os.path.join(data_path, matching_png))

# Convert RGB image to numpy array
rgb_image = np.array(rgb_image)

print("RGB Shape:", rgb_image.shape)

# Display images
plt.figure(figsize=(10,5))

# SAR Image
plt.subplot(1,2,1)
plt.imshow(sar_image, cmap='gray')
plt.title("SAR Image")
plt.axis('off')

# RGB Image
plt.subplot(1,2,2)
plt.imshow(rgb_image)
plt.title("Optical RGB Image")
plt.axis('off')

plt.show()