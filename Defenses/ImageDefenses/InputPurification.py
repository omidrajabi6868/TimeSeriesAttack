import torch

class FeatureDistillation(torch.nn.Module):
    def __init__(self, scale=30.0, block=8):
        super().__init__()
        self.scale = scale
        self.block = block

    def forward(self, x):
        """
        x: (B, C, H, W), float [0,1]
        """

        device = x.device
        B, C, H, W = x.shape

        dct_mat = self.build_dct_matrix(device)
        qtable = self.get_fd_table(device, self.scale)

        # scale to JPEG range
        x = x * 255.0

        # extract blocks
        blocks = x.unfold(2, self.block, self.block).unfold(3, self.block, self.block)
        # (B,C,H8,W8,8,8)

        # reshape for vectorized processing
        B, C, Hb, Wb, _, _ = blocks.shape

        blocks = blocks.contiguous().view(-1, self.block, self.block)
        # (B*C*Hb*Wb, 8, 8)

        # DCT
        freq = self.dct2(blocks, dct_mat)

        # quantization (FD core)
        quant = torch.round(freq / qtable)

        dequant = quant * qtable

        # IDCT
        rec = self.idct2(dequant, dct_mat)

        # reshape back
        rec = rec.view(B, C, Hb, Wb, self.block, self.block)

        # reconstruct image
        rec = rec.permute(0,1,2,4,3,5)
        rec = rec.reshape(B, C, H, W)

        return torch.clamp(rec / 255.0, 0, 1) 
    
    @staticmethod
    def build_dct_matrix(device):
        N = self.block
        dct = torch.zeros((N, N), device=device)

        for k in range(N):
            for n in range(N):
                if k == 0:
                    dct[k, n] = 1 / (N ** 0.5)
                else:
                    dct[k, n] = (2 / N) ** 0.5 * torch.cos(
                        (torch.pi * (2*n + 1) * k) / (2 * N)
                    )
        return dct

    @staticmethod
    def image_to_blocks(x):
        """
        x: (B, C, H, W)
        returns: (B, C, H/8, W/8, 8, 8)
        """

        B, C, H, W = x.shape

        blocks = x.unfold(2, self.block, self.block).unfold(3, self.block, self.block)

        return blocks
    
    @staticmethod
    def dct2(dct_mat):

        # left multiply
        tmp = torch.matmul(dct_mat, self.block)

        # right multiply
        return torch.matmul(tmp, dct_mat.T)

    @staticmethod
    def idct2(dct_mat):
        tmp = torch.matmul(dct_mat.T, self.block)
        return torch.matmul(tmp, dct_mat)

    @staticmethod
    def get_fd_table(device):
        jpeg = torch.tensor([
            [16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,36,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99],
        ], dtype=torch.float32, device=device)

        return jpeg * self.scale
