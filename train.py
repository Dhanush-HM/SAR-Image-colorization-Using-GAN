import torch
import os
import glob
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_loader import SAR2RGBDataset
from generator import Generator
from discriminator import Discriminator


# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using Device:", device)


# -------------------------
# Dataset
# -------------------------
data_path = r"datasets/patch_SAR_OPT_SQUARE"

dataset = SAR2RGBDataset(data_path)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)


# -------------------------
# Models
# -------------------------
generator = Generator().to(device)

discriminator = Discriminator().to(device)


# -------------------------
# Loss Functions
# -------------------------
adversarial_loss = nn.MSELoss()

l1_loss = nn.L1Loss()


# -------------------------
# Optimizers
# -------------------------
optimizer_G = optim.Adam(
    generator.parameters(),
    lr=0.0003,
    betas=(0.5, 0.999)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=0.0001,
    betas=(0.5, 0.999)
)
# -------------------------
# Load Latest Checkpoint
# -------------------------

start_epoch = 0

checkpoint_files = glob.glob("checkpoint_epoch_*.pth")

if checkpoint_files:

    latest_checkpoint = max(
        checkpoint_files,
        key=os.path.getctime
    )

    print(f"Loading: {latest_checkpoint}")

    checkpoint = torch.load(
        latest_checkpoint,
        map_location=device
    )

    generator.load_state_dict(
        checkpoint["generator_state_dict"]
    )

    discriminator.load_state_dict(
        checkpoint["discriminator_state_dict"]
    )

    optimizer_G.load_state_dict(
        checkpoint["optimizer_G_state_dict"]
    )

    optimizer_D.load_state_dict(
        checkpoint["optimizer_D_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    print(f"Resuming from epoch {start_epoch}")

else:

    print("No checkpoint found. Starting fresh training.")


# -------------------------
# Training Loop
# -------------------------
epochs = 10

for epoch in range(start_epoch, start_epoch + epochs):

    for batch_idx, (sar, real_rgb) in enumerate(dataloader):

        sar = sar.to(device)
        real_rgb = real_rgb.to(device)

        # -------------------------
        # Train Generator
        # -------------------------
        optimizer_G.zero_grad()

        # Generate fake RGB
        fake_rgb = generator(sar)

        # Discriminator prediction
        pred_fake = discriminator(sar, fake_rgb)

        # Generator adversarial loss
        valid = torch.ones_like(pred_fake).to(device)

        g_adv = adversarial_loss(pred_fake, valid)

        # L1 reconstruction loss
        g_l1 = l1_loss(fake_rgb, real_rgb)

        # Total generator loss
        g_loss = g_adv + 50 * g_l1

        g_loss.backward()

        optimizer_G.step()


        # -------------------------
        # Train Discriminator
        # -------------------------
        optimizer_D.zero_grad()

        # Real prediction
        pred_real = discriminator(sar, real_rgb)

        valid = torch.ones_like(pred_real).to(device)

        real_loss = adversarial_loss(pred_real, valid)

        # Fake prediction
        pred_fake = discriminator(sar, fake_rgb.detach())

        fake = torch.zeros_like(pred_fake).to(device)

        fake_loss = adversarial_loss(pred_fake, fake)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()

        optimizer_D.step()


        # -------------------------
        # Print Progress
        # -------------------------
        if batch_idx % 50 == 0:

            print(
                f"Epoch [{epoch+1}/{start_epoch + epochs}]"
                f"Batch [{batch_idx}/{len(dataloader)}] "
                f"D Loss: {d_loss.item():.4f} "
                f"G Loss: {g_loss.item():.4f}"
            )
    # -------------------------
    # Save Checkpoint Every 10 Epochs
    # -------------------------

    if (epoch + 1) % 10 == 0:

        checkpoint = {

            "epoch": epoch + 1,

            "generator_state_dict": generator.state_dict(),

            "discriminator_state_dict": discriminator.state_dict(),

            "optimizer_G_state_dict": optimizer_G.state_dict(),

            "optimizer_D_state_dict": optimizer_D.state_dict()
        }

        torch.save(
            checkpoint,
            f"checkpoint_epoch_{epoch+1}.pth"
        )

        print(f"\nCheckpoint saved at epoch {epoch+1}")

