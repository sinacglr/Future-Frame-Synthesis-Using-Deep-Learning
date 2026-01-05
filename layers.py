import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class CrossConvolutionalLayer(nn.Module):
    def __init__(self, kernel_size=config.KERNEL_SIZE, padding=config.PADDING):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, feature_maps, kernels):
        B, C, H, W = feature_maps.shape

        x = feature_maps.reshape(1, B*C, H, W)
        w = kernels.reshape(B*C, 1, self.kernel_size, self.kernel_size)

        out = F.conv2d(x, w, padding=self.padding, groups=B*C)
        return out.reshape(B, C, H, W)