import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def vae_loss(x, x_hat, mu, logvar, beta=2):
    
    reconstruction_loss = F.mse_loss(x_hat, x)

    kl = -0.5*torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return reconstruction_loss + beta * kl


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2)*(-math.log(10000)/d_model))

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TemporalTransformerVAE(nn.Module):
    def __init__(
        self,
        seq_len=24,
        input_dim=6,
        d_model=64,
        latent_dim=8,
        nhead=4,
        num_layers=2
    ):
        
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim

        # signal embedding
        self.input_proj = nn.Linear(input_dim, d_model)

        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, d_model)
        )

        self.pos_enc  = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.encoder  = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # latent projection
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # latent → decoder embedding
        self.latent_to_dmodel = nn.Linear(latent_dim, d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, input_dim)

    def encode(self, x):
        x = self.input_proj(x)

        B = x.shape[0]

        CLS = self.cls_token.expand(B, -1, -1)

        x = torch.cat([CLS, x], dim=1)

        x = self.pos_enc(x)

        h = self.encoder(x)

        cls_out = h[:, 0]

        mu = self.fc_mu(cls_out)
        logvar = self.fc_logvar(cls_out)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.latent_to_dmodel(z)

        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)

        h = self.pos_enc(h)

        h = self.decoder(h)

        x_hat = self.output_proj(h)

        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar
