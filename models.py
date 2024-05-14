
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class AE_model(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super().__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 3, 1, 1),          # B,  32, 64, 64
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  32,  32
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  16,  16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  128,  8,  8
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1),          # B,  128,  4,  4
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, 128),             # B, 128
            nn.ReLU(True),
            nn.Linear(128, z_dim)          # B, zdim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),               # B, 128
            nn.ReLU(True),
            nn.Linear(128, 256),                  # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4),      # B,  128,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),      # B,  128,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  64,  16,  16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 64, 64
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 3, 1, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        # latent is returned to facilitate perturbation
        latent = self._encode(x)
        x_recon = self._decode(latent)
        return x_recon, latent

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class BetaVAE_model(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super().__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 3, 1, 1),          # B,  32, 64, 64
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  32,  32
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  16,  16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),          # B,  128,  8,  8
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1),          # B,  128,  4,  4
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, 128),             # B, 128
            nn.ReLU(True),
            nn.Linear(128, z_dim*2)          # B, zdim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128),               # B, 128
            nn.ReLU(True),
            nn.Linear(128, 256),                  # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4),      # B,  128,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),      # B,  128,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # B,  64,  16,  16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 64, 64
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 3, 1, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)
