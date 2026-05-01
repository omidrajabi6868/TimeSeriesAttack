import torch
import torch.nn as nn

# ========================
# RevIN (Reversible Instance Norm)
# ========================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x, mode):
        if mode == 'norm':
            self.mean = x.mean(dim=1, keepdim=True)
            self.std = x.std(dim=1, keepdim=True) + self.eps
            x = (x - self.mean) / self.std
            return x * self.gamma + self.beta

        elif mode == 'denorm':
            x = (x - self.beta) / (self.gamma + self.eps)
            return x * self.std + self.mean


# ========================
# Patch Embedding
# ========================
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):
        # x: [B*C, L]
        B, L = x.shape

        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # [B*C, num_patches, patch_len]

        return self.proj(patches)


# ========================
# Full PatchTST
# ========================
class PatchTST(nn.Module):
    def __init__(self,
                 input_len,
                 pred_len,
                 num_vars,
                 patch_len=16,
                 stride=8,
                 d_model=128,
                 n_heads=4,
                 n_layers=3,
                 dropout=0.1):

        super().__init__()

        self.num_vars = num_vars
        self.pred_len = pred_len

        # RevIN
        self.revin = RevIN(num_vars)

        # Patch embedding (channel independent)
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model)

        # Compute number of patches
        self.num_patches = (input_len - patch_len) // stride + 1

        # Positional embedding
        self.pos_emb = nn.Parameter(
            torch.randn(1, self.num_patches, d_model)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Head: flatten patches
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_patches * d_model, pred_len)
        )

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape

        # RevIN normalize
        x = self.revin(x, 'norm')

        # Channel independence
        x = x.permute(0, 2, 1)        # [B, C, L]
        x = x.reshape(B * C, L)       # [B*C, L]

        # Patch embedding
        x = self.patch_embed(x)       # [B*C, num_patches, d_model]

        # Add positional encoding
        x = x + self.pos_emb

        # Transformer
        x = self.encoder(x)

        # Prediction
        x = self.head(x)              # [B*C, pred_len]

        # Restore shape
        x = x.view(B, C, self.pred_len)
        x = x.permute(0, 2, 1)        # [B, pred_len, C]

        # RevIN denormalize
        x = self.revin(x, 'denorm')

        return x