import torch
import torch.nn as nn


# -------------------------
# Encoder Block
# -------------------------
class EncoderBlock(nn.Module):

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
# Decoder Block
# -------------------------
class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# -------------------------
# Generator (U-Net)
# -------------------------
class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder
        self.e1 = EncoderBlock(1, 64, normalize=False)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.d1 = DecoderBlock(512, 256, dropout=True)
        self.d2 = DecoderBlock(512, 128, dropout=True)
        self.d3 = DecoderBlock(256, 64)

        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Tanh()
        )

    def forward(self, x):

        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder + Skip Connections
        d1 = self.d1(b)

        d1 = torch.cat([d1, e3], dim=1)

        d2 = self.d2(d1)

        d2 = torch.cat([d2, e2], dim=1)

        d3 = self.d3(d2)

        d3 = torch.cat([d3, e1], dim=1)

        output = self.final(d3)

        return output


# -------------------------
# Test Generator
# -------------------------
if __name__ == "__main__":

    # Create random SAR batch
    x = torch.randn(4, 1, 112, 112)

    # Create model
    model = Generator()

    # Generate output
    y = model(x)

    print("Input Shape :", x.shape)
    print("Output Shape:", y.shape)
    print("Output Min  :", y.min().item())
    print("Output Max  :", y.max().item())