"""U-Net réduit (peu de canaux) pour un entraînement raisonnable sur CPU."""

import torch
from torch import nn


def bloc_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetLeger(nn.Module):
    """U-Net à 4 niveaux, canaux réduits (16/32/64/128) pour rester
    raisonnable sur une machine sans GPU dédié."""

    def __init__(self, in_channels: int, n_classes: int, canaux=(16, 32, 64, 128)):
        super().__init__()
        c1, c2, c3, c4 = canaux

        self.enc1 = bloc_conv(in_channels, c1)
        self.enc2 = bloc_conv(c1, c2)
        self.enc3 = bloc_conv(c2, c3)
        self.pool = nn.MaxPool2d(2)

        self.milieu = bloc_conv(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = bloc_conv(c3 + c3, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = bloc_conv(c2 + c2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = bloc_conv(c1 + c1, c1)

        self.sortie = nn.Conv2d(c1, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.milieu(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.sortie(d1)
