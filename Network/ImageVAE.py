import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ImageVAE(nn.Module):
    def __init__(
        self,
        image_channels=3,
        image_size=(128, 128),
        latent_dim=128,
        hidden_dims=None,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        self.image_channels = image_channels
        self.image_size = image_size
        self.latent_dim = latent_dim

        encoder_layers = []
        in_channels = image_channels
        for h_dim in hidden_dims:
            encoder_layers.append(_ConvBlock(in_channels, h_dim, stride=2))
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, image_channels, image_size[0], image_size[1])
            encoder_out = self.encoder(dummy)

        self._feature_shape = encoder_out.shape[1:]
        flattened_dim = int(torch.prod(torch.tensor(self._feature_shape)).item())

        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, flattened_dim)

        decoder_layers = []
        reversed_dims = hidden_dims[::-1]
        for idx in range(len(reversed_dims) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        reversed_dims[idx],
                        reversed_dims[idx + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(reversed_dims[idx + 1]),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(reversed_dims[idx + 1], reversed_dims[idx + 1], kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(reversed_dims[idx + 1]),
                    nn.LeakyReLU(inplace=True),
                )
            )

        self.decoder = nn.Sequential(*decoder_layers)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                reversed_dims[-1],
                reversed_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(reversed_dims[-1]),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(reversed_dims[-1], image_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        features = self.encoder(x)
        flattened = torch.flatten(features, start_dim=1)
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        decoded = self.decoder_input(z)
        decoded = decoded.view(-1, *self._feature_shape)
        decoded = self.decoder(decoded)
        x_hat = self.final_layer(decoded)

        target_h, target_w = self.image_size
        if x_hat.shape[-2:] != (target_h, target_w):
            x_hat = nn.functional.interpolate(x_hat, size=(target_h, target_w), mode='bilinear', align_corners=False)

        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
