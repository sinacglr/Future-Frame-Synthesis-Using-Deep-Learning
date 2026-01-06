import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import CrossConvolutionalLayer
import config

BN = nn.BatchNorm2d

class FutureFramePredictionModel(nn.Module):
    def __init__(self, z_dim=config.Z_DIM):
        super().__init__()
        self.z_dim = z_dim

        self.enc = nn.Sequential(
            nn.Conv2d(6,96,5,1,2), BN(96), nn.ReLU(True),
            nn.Conv2d(96,96,5,1,2), BN(96), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(96,128,5,1,2), BN(128), nn.ReLU(True),
            nn.Conv2d(128,128,5,1,2), BN(128), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(128,256,5, 1, 2), BN(256), nn.ReLU(True),
            nn.Conv2d(256,256,5, 1, 2), BN(256), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((5,5))
        )

        self.z_mu = nn.Linear(256*25, z_dim)
        self.z_logvar = nn.Linear(256*25, z_dim)

        self.k_dec = nn.Sequential(nn.Linear(z_dim, 128*config.KERNEL_SIZE*config.KERNEL_SIZE))

        self.img_enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3,64,5,1,2), BN(64), nn.ReLU(True), 
                          nn.Conv2d(64,64,5,1,2), BN(64), nn.ReLU(True), nn.MaxPool2d(2), 
                          nn.Conv2d(64,64,5,1,2), BN(64), nn.ReLU(True), 
                          nn.Conv2d(64,32,5,1,2), BN(32), nn.ReLU(True), nn.MaxPool2d(2)) 
            for _ in range(4)
        ])

        self.cross = CrossConvolutionalLayer()

        self.dec = nn.Sequential(
            nn.Conv2d(128,128,9,1,4), BN(128), nn.ReLU(True),
            nn.Conv2d(128,128,1), BN(128), nn.ReLU(True),
            nn.Conv2d(128,3,1)
        )

    def encode(self, i1, v):
        x = self.enc(torch.cat([i1,v],1)).view(i1.size(0), -1)
        return self.z_mu(x), self.z_logvar(x)

    def forward(self, i1, v=None, temp=1.0, z_input=None):
        if v is not None:
            mu, lv = self.encode(i1, v)
            z = mu + torch.randn_like(mu)*torch.exp(0.5*lv)
        else:
            mu, lv = None, None
            if z_input is not None: 
                z = z_input
            else: 
                z = torch.randn(i1.size(0), self.z_dim, device=i1.device) * temp

        k_raw = self.k_dec(z).view(-1, 128, config.KERNEL_SIZE*config.KERNEL_SIZE)
        k = F.softmax(k_raw, dim=-1)
        k = k.view(-1, 4, 32, config.KERNEL_SIZE, config.KERNEL_SIZE)

        feats = [self.img_enc[i](F.interpolate(i1, size=(s,s))) for i,s in enumerate([256,128,64,32])]

        fused = torch.cat([F.interpolate(self.cross(feats[s], k[:,s]), size=(64,64)) for s in range(4)], 1)

        v_pred = F.interpolate(self.dec(fused), size=(128,128), mode='bilinear', align_corners=False)

        return (v_pred, mu, lv) if v is not None else v_pred
