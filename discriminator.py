import torch
import torch.nn as nn


# -------------------------
# Discriminator Block
# -------------------------
class DiscriminatorBlock(nn.Module):

    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        ]

        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# -------------------------
# PatchGAN Discriminator
# -------------------------
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            # Input = SAR(1) + RGB(3)
            DiscriminatorBlock(4, 64, normalize=False),

            DiscriminatorBlock(64, 128),

            DiscriminatorBlock(128, 256),

            nn.Conv2d(
                256,
                1,
                kernel_size=4,
                stride=1,
                padding=1
            )
        )

    def forward(self, sar, rgb):

        # Concatenate SAR + RGB
        x = torch.cat([sar, rgb], dim=1)

        return self.model(x)


# -------------------------
# Test Discriminator
# -------------------------
if __name__ == "__main__":

    # Fake SAR batch
    sar = torch.randn(4, 1, 112, 112)

    # Fake RGB batch
    rgb = torch.randn(4, 3, 112, 112)

    # Create discriminator
    model = Discriminator()

    # Forward pass
    output = model(sar, rgb)

    print("SAR Shape :", sar.shape)
    print("RGB Shape :", rgb.shape)

    print("\nDiscriminator Output Shape:")
    print(output.shape)