import torch

import torch.nn as nn
import torch.nn.functional as F
from spatial_broadcast import SpatialBroadcastLayer2d
from torch.testing import randn_like


class Encoder(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.pad = nn.ConstantPad3d((2, 3, 9, 10, 2, 3), 0)
        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(16),

            nn.Conv3d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(64, 64, 3, 1, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(128),

            nn.Conv3d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(256),

        )

        self.dense = nn.Linear(256, embedding_size)
        self.dense_var = nn.Linear(256, embedding_size)

    def forward(self, img):
        batch_size = img.shape[0]
        img = self.pad(img)
        conv_img = self.conv(img)
        avg_channel = conv_img.view(batch_size, 256, -1).mean(dim=2)
        # avg_channel = F.dropout(avg_channel, p=0.1)
        mean = self.dense(avg_channel)
        log_var = self.dense_var(avg_channel)
        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.dense = nn.Linear(embedding_size, 256)

        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(256, 256, 3, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose3d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(128),

            nn.ConvTranspose3d(128, 128, 3, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(64),

            nn.ConvTranspose3d(64, 64, 3, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(32),

            nn.ConvTranspose3d(32, 32, 3, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm3d(16),

            nn.ConvTranspose3d(16, 16, 3, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, 3, 2, 1, 1)
        )

    def forward(self, latent):
        batch_size = latent.shape[0]
        avg_channel = self.dense(latent)
        # avg_channel = F.dropout(avg_channel, p=0.1)
        avg_channel = avg_channel[:, :, None,
                      None, None].expand(batch_size, 256, 3, 4, 3) * 1
        rec = self.deconv(avg_channel)

        # self.pad = nn.ConstantPad3d((2, 3, 9, 10, 2, 3), 0)
        rec = rec[:, :, 2:-3, 9:-10, 2:-3]
        return rec


class Encoder2d(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.pad = nn.ConstantPad2d((2, 3, 9, 10), 0)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.dense = nn.Linear(256, embedding_size)
        self.dense_var = nn.Linear(256, embedding_size)

    def forward(self, img):
        batch_size = img.shape[0]
        img = self.pad(img)
        conv_img = self.conv(img)
        avg_channel = conv_img.view(batch_size, 256, -1).mean(dim=2)
        # avg_channel = F.dropout(avg_channel, p=0.1)
        mean = self.dense(avg_channel)
        log_var = self.dense_var(avg_channel)
        return mean, log_var


class Decoder2d(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.dense = nn.Linear(embedding_size, 256)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.ConvTranspose2d(128, 128, 3, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, 3, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 32, 3, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.ConvTranspose2d(16, 16, 3, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1)
        )

    def forward(self, latent):
        batch_size = latent.shape[0]
        avg_channel = self.dense(latent)
        # avg_channel = F.dropout(avg_channel, p=0.1)
        avg_channel = avg_channel[:, :, None, None].expand(
            batch_size, 256, 3, 4) * 1  # why?
        rec = self.deconv(avg_channel)

        # self.pad = nn.ConstantPad3d((2, 3, 9, 10, 2, 3), 0)
        rec = rec[:, :, 2:-3, 9:-10]
        return rec


class SpatialBroadcastDecoder2d(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()

        self.spatial_broadcast = SpatialBroadcastLayer2d(
            embedding_size, 91, 109)
        self.conv = nn.Sequential(
            nn.Conv2d(embedding_size + 2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )

    def forward(self, latent):
        return self.conv(self.spatial_broadcast(latent))


class VAE(nn.Module):
    def __init__(self, embedding_dim=128, input_2d=True, reparam=True,
                 use_spatial_broadcast=True):
        super().__init__()
        self.reparam = reparam
        if input_2d:
            self.encoder = Encoder2d(embedding_dim)
            if use_spatial_broadcast:
                self.decoder = SpatialBroadcastDecoder2d(embedding_dim)
            else:
                self.decoder = Decoder2d(embedding_dim)
        else:
            self.encoder = Encoder(embedding_dim)
            self.decoder = Decoder(embedding_dim)

    def forward(self, img):
        mean, log_var = self.encoder(img)
        penalty = gaussian_kl(mean, log_var)
        if self.training and self.reparam:
            latent = reparameterize(mean, log_var)
        else:
            latent = mean
        return self.decoder(latent), penalty


def gaussian_kl(mean, log_var):
    return .5 * torch.sum(mean ** 2 + torch.exp(log_var)
                          - log_var - 1) / mean.shape[0]


def masked_mse(pred, target, mask):
    diff = pred - target
    mask = mask ^ 1
    mask = mask[None, None, ...]
    diff.masked_fill_(mask, 0.)
    return torch.sum(diff ** 2) / diff.shape[0]


def reparameterize(mean, log_var):
    eps = randn_like(mean)
    return mean + torch.exp(log_var / 2) * eps
