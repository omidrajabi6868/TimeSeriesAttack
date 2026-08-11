import torch
import torch.nn as nn

# Implementation from https://github.com/timctho/unet-pytorch/
class IUNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(IUNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class IUNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(IUNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.conv3 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class IUNet(torch.nn.Module):
    def __init__(self):
        super(IUNet, self).__init__()
        self.down_block1 = IUNet_down_block(3, 16, False)
        self.down_block2 = IUNet_down_block(16, 32, True)
        self.down_block3 = IUNet_down_block(32, 64, True)
        self.down_block4 = IUNet_down_block(64, 128, True)
        self.down_block5 = IUNet_down_block(128, 256, True)
        self.down_block6 = IUNet_down_block(256, 512, True)
        self.down_block7 = IUNet_down_block(512, 1024, True)

        self.mid_conv1 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(1024)
        self.mid_conv2 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(1024)

        self.up_block1 = IUNet_up_block(512, 1024, 512)
        self.up_block2 = IUNet_up_block(256, 512, 256)
        self.up_block3 = IUNet_up_block(128, 256, 128)
        self.up_block4 = IUNet_up_block(64, 128, 64)
        self.up_block5 = IUNet_up_block(32, 64, 32)
        self.up_block6 = IUNet_up_block(16, 32, 16)

        self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm2d(16)
        self.last_conv2 = torch.nn.Conv2d(16, 3, 1, padding=0)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)
        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))
        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x


"""Original U-Net"""


class UNet_down_block(torch.nn.Module):
    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.max_pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet_up_block(torch.nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()
        self.up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = torch.nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(torch.nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_block1 = UNet_down_block(3, 64, False)
        self.down_block2 = UNet_down_block(64, 128, True)
        self.down_block3 = UNet_down_block(128, 256, True)
        self.down_block4 = UNet_down_block(256, 512, True)
        self.down_block5 = UNet_down_block(512, 1024, True)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)

        self.last_conv = torch.nn.Conv2d(64, 3, 1, padding=0)

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        x = self.up_block1(self.x4, self.x5)
        x = self.up_block2(self.x3, x)
        x = self.up_block3(self.x2, x)
        x = self.up_block4(self.x1, x)
        x = self.last_conv(x)
        return x


class ParameterRender(torch.nn.Module):
    def __init__(self):
        super(ParameterRender, self).__init__()

        self.down_block1 = UNet_down_block(3, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 32, True)
        self.down_block4 = UNet_down_block(32, 64, True)

        self.up_block1 = UNet_up_block(32, 64, 64)
        self.up_block2 = UNet_up_block(32, 64, 32)
        self.up_block3 = UNet_up_block(16, 32, 16)

        self.last_conv = torch.nn.Conv2d(16, 3, 1, padding=0)

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        x = self.up_block1(self.x3, self.x4)
        x = self.up_block2(self.x2, x)
        x = self.up_block3(self.x1, x)
        x = self.last_conv(x)

        return x


class VisualTransformerEncoder(nn.Module):
    def __init__(self, input_size, patch_size, num_patches, emb_dim, num_heads, num_layers, channel):
        super(VisualTransformerEncoder, self).__init__()
        self.input_size = input_size
        self.channel = channel
        self.patch_size = patch_size
        self.num_patches = num_patches
        # self.patch_emb = nn.Conv2d(1, emb_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_emb = nn.Linear((patch_size ** 2) * channel, emb_dim)
        self.cls_emb = nn.Parameter(torch.rand(1, 1, emb_dim))
        self.positional_emb = nn.Parameter(torch.rand(1, num_patches + 1, emb_dim))

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        x = x.view(len(x), x.shape[2] // self.patch_size, x.shape[3] // self.patch_size, self.channel, self.patch_size,
                   self.patch_size)
        x = x.view(len(x), self.num_patches, (self.patch_size ** 2) * self.channel)
        x = self.patch_emb(x)
        cls_token = self.cls_emb.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x += self.positional_emb
        for block in self.transformer_blocks:
            x = block(x)
        x = x[:, :, :]
        return x


class VisualTransformerDecoder(nn.Module):
    def __init__(self, patch_size, num_patches,
                 emb_dim, num_heads, num_layers, channel):
        super(VisualTransformerDecoder, self).__init__()
        self.patch_size = patch_size
        self.channel = channel

        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=emb_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])

        self.fc2 = nn.Linear(emb_dim, (self.patch_size ** 2) * self.channel)

    def forward(self, encoder_output):
        for block in self.transformer_blocks:
            encoder_output = block(encoder_output)
        x = self.fc2(encoder_output)

        return x[:, :, :]


class VisualTransformerAutoencoder(nn.Module):
    def __init__(self, image_size, patch_size, emb_dim, num_heads, num_layers, num_patches, output_channels):
        super(VisualTransformerAutoencoder, self).__init__()
        self.num_patches = num_patches
        self.encoder = VisualTransformerEncoder(image_size, patch_size, num_patches, emb_dim, num_heads, num_layers,
                                                output_channels)
        self.decoder = VisualTransformerDecoder(patch_size, num_patches, emb_dim, num_heads, num_layers,
                                                output_channels)

    def forward(self, x):
        encoder_output = self.encoder(x)
        x = encoder_output + self.encoder.positional_emb
        reconstruction = self.decoder(x)[:, 1:, :]
        reconstruction = reconstruction.view(len(reconstruction), self.encoder.input_size // self.encoder.patch_size,
                                             self.encoder.input_size // self.encoder.patch_size, self.encoder.channel,
                                             self.encoder.patch_size, self.encoder.patch_size)
        reconstruction = reconstruction.view(
            (len(reconstruction), self.encoder.channel, self.encoder.input_size, self.encoder.input_size))
        return reconstruction