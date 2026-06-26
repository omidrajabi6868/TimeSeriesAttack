import torch
import math


class FeatureDistillation(torch.nn.Module):
    """
    Paper-aligned implementation of:
    "Feature Distillation: DNN-Oriented JPEG Compression Against Adversarial Examples"
    """

    def __init__(self, std_map, block=8, QS=30.0, S1=50.0, S2=10.0):
        super().__init__()

        self.block = block

        # -------------------------------------------------
        # Step 1: GLOBAL DEFENSIVE QUANTIZATION STRENGTH
        # QS controls adversarial suppression level
        # -------------------------------------------------
        self.QS = QS

        # Step 2: band-level quantization parameters
        self.S1 = S1  # Malicious Defense band (strong compression)
        self.S2 = S2  # Accuracy Sensitive band (weak compression)

        self.register_buffer("std_map", std_map)
        self.register_buffer("jpeg_table", self._build_jpeg_table())

        # -------------------------------------------------
        # Paper Step 2: frequency importance + band split
        # -------------------------------------------------
        self.register_buffer("freq_importance", self._build_importance(std_map))
        self.register_buffer("qs_table", self._build_qs_table(self.freq_importance))

    # =====================================================
    # FORWARD (ONE-PASS DEFENSE PIPELINE)
    # =====================================================
    def forward(self, x):
        device = x.device
        B, C, H, W = x.shape
        N = self.block

        dct_mat = self.build_dct_matrix(device, N)

        # JPEG scaling assumption
        x = x * 255.0

        # blockify
        blocks = x.unfold(2, N, N).unfold(3, N, N)
        B, C, Hb, Wb, _, _ = blocks.shape

        blocks = blocks.contiguous().view(-1, N, N)

        # DCT
        freq = self.dct2(blocks, dct_mat)

        # =====================================================
        # PAPER CORE: QS × Q(i,j)
        # =====================================================

        q_band = self.qs_table.to(device)

        # FULL PAPER QUANTIZATION MODEL:
        # (Step 1 global suppression + Step 2 frequency adaptation)
        q = self.QS * q_band

        freq = torch.round(freq / q) * q

        # IDCT
        rec = self.idct2(freq, dct_mat)

        # reshape back
        rec = rec.view(B, C, Hb, Wb, N, N)
        rec = rec.permute(0, 1, 2, 4, 3, 5)
        rec = rec.reshape(B, C, H, W)

        return torch.clamp(rec / 255.0, 0, 1)

    # =====================================================
    # STEP 2: FREQUENCY IMPORTANCE (δi,j)
    # =====================================================
    def _build_importance(self, std_map):
        """
        δi,j estimation from dataset statistics.
        Paper: standard deviation over DCT coefficients.
        """
        importance = std_map / (std_map.mean() + 1e-6)

        # normalize to [0,1]
        importance = importance / (importance.max() + 1e-6)

        return importance

    # =====================================================
    # STEP 2: AS / MD BAND SPLIT (PAPER-STYLE)
    # =====================================================
    def _build_qs_table(self, importance):
        """
        Paper:
        QSi,j = S2 if frequency is important (AS band)
                S1 otherwise (MD band)
        """

        # rank-based split (closer to paper than median)
        flat = importance.view(-1)

        # top 50% = Accuracy Sensitive (AS band)
        threshold = torch.quantile(flat, 0.5)

        qs = torch.where(
            importance >= threshold,
            torch.tensor(self.S2),
            torch.tensor(self.S1)
        )

        return qs

    # =====================================================
    # DCT UTILITIES
    # =====================================================
    @staticmethod
    def build_dct_matrix(device, block):
        N = block
        dct = torch.zeros((N, N), device=device)

        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct[k, n] = 1 / math.sqrt(N)
                else:
                    dct[k, n] = math.sqrt(2 / N) * math.cos(
                        math.pi * (2 * n + 1) * k / (2 * N)
                    )
        return dct

    @staticmethod
    def dct2(x, dct):
        return dct @ x @ dct.t()

    @staticmethod
    def idct2(x, dct):
        return dct.t() @ x @ dct

    # =====================================================
    # JPEG BASE TABLE (REFERENCE ONLY)
    # =====================================================
    @staticmethod
    def _build_jpeg_table():
        return torch.tensor([
            [16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,36,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99],
        ], dtype=torch.float32)