import torch
import torch.nn as nn
import torch.nn.functional as F

class Scissors(nn.Module):
    def __init__(self, trainable_weights=False):
        super().__init__()
        # 初始化核心修正点：添加 clone() 断开内存共享
        default_weights = torch.tensor([
            [[0.025,0.035,0.035,0.035,0.025],
             [0.035,0.050,0.070,0.050,0.035],
             [0.035,0.070,0.000,0.070,0.035],
             [0.035,0.050,0.070,0.050,0.035],
             [0.025,0.035,0.035,0.035,0.025]]
        ], dtype=torch.float16)  # (1,5,5)
        
        # 关键修复：通过 clone() 创建独立内存副本
        default_weights = default_weights.expand(3,5,5).clone()  # 形状 (3,5,5)
        
        if trainable_weights:
            self.weights = nn.Parameter(default_weights)
        else:
            self.register_buffer('weights', default_weights)

    def forward(self, x):
        B, C, H, W = x.shape
        assert self.weights.shape[0] == C, \
            f"权重通道数 {self.weights.shape[0]} vs 输入通道 {C} 不匹配"
        
        padded = F.pad(x, (2,2,2,2), mode='replicate')
        windows = padded.unfold(2,5,1).unfold(3,5,1)  # (B,C,H,W,5,5)
        flipped = torch.flip(windows, dims=[4,5])
        
        # 使用 reshape 保持内存连续性
        weighted_diff = torch.abs(windows - flipped) * self.weights.view(1,C,1,1,5,5)
        diff_map = torch.mean(weighted_diff, dim=(4,5))
        
        return torch.cat([x, torch.mean(diff_map, dim=1, keepdim=True)], dim=1)