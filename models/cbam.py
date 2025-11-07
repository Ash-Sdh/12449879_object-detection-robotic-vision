import torch, torch.nn as nn, torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_ch, r=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, in_ch // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // r, in_ch, bias=False)
        )
    def forward(self, x):
        b, c, h, w = x.size()
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx  = F.adaptive_max_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k//2, bias=False)
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx  = torch.max(x, dim=1, keepdim=True).values
        m = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * m

class CBAM(nn.Module):
    def __init__(self, ch, r=16, k=7):
        super().__init__()
        self.ca = ChannelAttention(ch, r)
        self.sa = SpatialAttention(k)
    def forward(self, x):
        return self.sa(self.ca(x))
