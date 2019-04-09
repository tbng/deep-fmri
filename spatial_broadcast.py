import numpy as np
from numpy.testing import assert_allclose
import torch
from torch import nn


class SpatialBroadcastLayer2D(nn.Module):

    def __init__(self, input_dim, h_dim, w_dim):
        super(SpatialBroadcastLayer2D, self).__init__()
        self.input_dim = input_dim
        self.h_dim = h_dim
        self.w_dim = w_dim
        h_grid, w_grid = torch.meshgrid(torch.linspace(-1, 1, h_dim),
                                        torch.linspace(-1, 1, w_dim))
        self.register_buffer("h_grid", h_grid.view((1, 1) + h_grid.shape))
        self.register_buffer("w_grid", w_grid.view((1, 1) + w_grid.shape))

    def forward(self, z):
        input_dim, h, w = self.input_dim, self.h_dim, self.w_dim
        batch_size = z.shape[0]
        assert z.shape[1] == input_dim

        # Tile z vectors across to match image size for sample in the
        # minibatch. The resulting shape is (N, D, H, W).
        z_grid = z.view(z.shape + (1, 1)).expand(-1, -1, h, w)

        # Concatenate along the channel dimension, output has shape:
        # (N, 2 + D, H, W)
        return torch.cat((self.h_grid.expand(batch_size, -1, -1, -1),
                          self.w_grid.expand(batch_size, -1, -1, -1),
                          z_grid), dim=1)


def test_spatial_broadcast():
    h, w = 3, 4
    batch_size = 2
    z_dim = 5
    z = np.arange(batch_size * z_dim).reshape(batch_size, z_dim)
    z = torch.FloatTensor(z)

    sb = SpatialBroadcastLayer2D(z_dim, 3, 4)
    sb_out = sb.forward(z)

    assert sb_out.shape == (2, 2 + 5, h, w)

    for j in range(w):
        assert_allclose(sb_out[0, 0, :, j], np.linspace(-1, 1, h), rtol=1e-5)
        assert_allclose(sb_out[1, 0, :, j], np.linspace(-1, 1, h), rtol=1e-5)

    for i in range(h):
        assert_allclose(sb_out[0, 1, i, :], np.linspace(-1, 1, w), rtol=1e-5)
        assert_allclose(sb_out[1, 1, i, :], np.linspace(-1, 1, w), rtol=1e-5)

    for zi in z[0, :].numpy():
        constant_zmap = sb_out[0, int(zi) + 2, :, :]
        assert_allclose(constant_zmap, int(zi) * np.ones_like(constant_zmap))


if __name__ == "__main__":
    test_spatial_broadcast()
