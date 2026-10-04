"""Compact 128px restoration, routing, and conditional sketch models."""
from __future__ import annotations

import torch
from torch import nn


def conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, 1),
                         nn.GroupNorm(min(8, out_channels), out_channels), nn.LeakyReLU(0.2, inplace=True))


class ConvAutoencoder(nn.Module):
    def __init__(self, base: int = 16, bottleneck: int = 48):
        super().__init__()
        self.encoder = nn.Sequential(conv_block(3, base), conv_block(base, base * 2, 2),
                                     conv_block(base * 2, base * 4, 2),
                                     conv_block(base * 4, bottleneck, 2))
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), conv_block(bottleneck, base * 4),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), conv_block(base * 4, base * 2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), conv_block(base * 2, base),
            nn.Conv2d(base, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(image))


class CorruptionClassifier(nn.Module):
    def __init__(self, base: int = 16, dropout: float = 0.1):
        super().__init__()
        self.features = nn.Sequential(conv_block(3, base, 2), conv_block(base, base * 2, 2),
                                      conv_block(base * 2, base * 4, 2), conv_block(base * 4, base * 4, 2),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(base * 4, 4))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(image))


class SoftMixture(nn.Module):
    """Branch 0 is identity; branches 1-3 are specialist restorers."""
    def __init__(self, gate: CorruptionClassifier, experts: list[ConvAutoencoder]):
        super().__init__()
        if len(experts) != 3:
            raise ValueError("Three trained experts required")
        self.gate = gate
        self.experts = nn.ModuleList(experts)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.gate(image), dim=1)
        branches = torch.stack([image] + [expert(image) for expert in self.experts], dim=1)
        reconstructed = (branches * weights[:, :, None, None, None]).sum(dim=1)
        return reconstructed, weights


class Down(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, norm: bool = True):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(input_channels, output_channels, 4, 2, 1),
                                   nn.GroupNorm(min(8, output_channels), output_channels) if norm else nn.Identity(),
                                   nn.LeakyReLU(0.2, inplace=True))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.block(image)


class Up(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dropout: float = 0):
        super().__init__()
        self.block = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1),
                                   nn.GroupNorm(min(8, output_channels), output_channels), nn.ReLU(inplace=True),
                                   nn.Dropout(dropout))

    def forward(self, image: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.block(image), skip), dim=1)


class SketchGenerator(nn.Module):
    def __init__(self, base: int = 16, embedding_dim: int = 8, dropout: float = 0.2):
        super().__init__()
        self.style = nn.Embedding(3, embedding_dim)
        self.down1 = Down(3 + embedding_dim, base, norm=False)
        self.down2 = Down(base, base * 2)
        self.down3 = Down(base * 2, base * 4)
        self.down4 = Down(base * 4, base * 8)
        self.down5 = Down(base * 8, base * 8)
        self.up1 = Up(base * 8, base * 8, dropout)
        self.up2 = Up(base * 16, base * 4, dropout)
        self.up3 = Up(base * 8, base * 2)
        self.up4 = Up(base * 4, base)
        self.final = nn.Sequential(nn.ConvTranspose2d(base * 2, 3, 4, 2, 1), nn.Tanh())

    def forward(self, image: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        emb = self.style(style_id.long()).view(image.size(0), -1, 1, 1)
        x = torch.cat([image, emb.expand(-1, -1, image.size(2), image.size(3))], dim=1)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        return self.final(u4)


class StylePatchDiscriminator(nn.Module):
    def __init__(self, base: int = 16, embedding_dim: int = 8):
        super().__init__()
        self.style = nn.Embedding(3, embedding_dim)
        self.network = nn.Sequential(Down(6 + embedding_dim, base, norm=False),
                                     Down(base, base * 2), Down(base * 2, base * 4),
                                     Down(base * 4, base * 8),
                                     nn.Conv2d(base * 8, 1, 3, padding=1))

    def forward(self, photo: torch.Tensor, sketch: torch.Tensor, style_id: torch.Tensor) -> torch.Tensor:
        emb = self.style(style_id.long()).view(photo.size(0), -1, 1, 1)
        image = torch.cat((photo, sketch, emb.expand(-1, -1, photo.size(2), photo.size(3))), dim=1)
        return self.network(image)
