#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedKANBlock(nn.Module):
    """
    增强型Kolmogorov-Arnold Network (KAN) 模块
    
    特点:
    1. 多头基函数学习
    2. 可配置的激活函数
    3. 残差连接
    4. 注意力机制
    5. 可配置的层数和维度
    """
    def __init__(self, in_channels, hidden_dim=128, heads=4, layers=5, 
                 dropout=0.2, activation='relu', use_attention=True):
        """
        初始化增强型KAN模块
        
        参数:
            in_channels: 输入特征的通道数
            hidden_dim: 隐藏层维度
            heads: 多头机制中的头数
            layers: KAN模块中的层数
            dropout: Dropout率
            activation: 激活函数类型 ('relu', 'gelu', 'silu', 'mish')
            use_attention: 是否使用注意力机制
        """
        super(EnhancedKANBlock, self).__init__()
        self.layers = layers
        self.heads = heads
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.use_attention = use_attention
        
        # 定义激活函数
        if activation == 'gelu':
            self.act_fn = nn.GELU()
        elif activation == 'silu':
            self.act_fn = nn.SiLU()
        elif activation == 'mish':
            self.act_fn = nn.Mish()
        else:
            self.act_fn = nn.ReLU(inplace=True)
        
        # 多头基函数参数 (每个头有自己的参数)
        self.alpha = nn.Parameter(torch.randn(heads, hidden_dim))
        self.beta = nn.Parameter(torch.randn(heads, hidden_dim))
        self.gamma = nn.Parameter(torch.randn(heads, hidden_dim))
        
        # 投影和层归一化
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.layer_norm = nn.LayerNorm([hidden_dim, 1, 1])
        
        # 中间层处理
        self.mid_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                self.act_fn,
                nn.Dropout2d(dropout)
            ) for _ in range(layers)
        ])
        
        # 注意力机制
        if use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, hidden_dim // 8, kernel_size=1),
                self.act_fn,
                nn.Conv2d(hidden_dim // 8, hidden_dim, kernel_size=1),
                nn.Sigmoid()
            )
        
        # 输出投影
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [B, C, H, W]
            
        返回:
            out: 增强后的特征 [B, C, H, W]
        """
        identity = x
        
        # 输入投影
        h = self.input_proj(x)
        
        # 应用多头KAN基函数和中间层
        for i in range(self.layers):
            # 创建多头基函数输出
            heads_output = []
            for head in range(self.heads):
                # 应用基函数: alpha * sin(beta * x + gamma)
                h_base = self.alpha[head].view(1, -1, 1, 1) * torch.sin(
                    self.beta[head].view(1, -1, 1, 1) * h + self.gamma[head].view(1, -1, 1, 1)
                )
                heads_output.append(h_base)
            
            # 合并多头输出 (取平均)
            h = torch.stack(heads_output).mean(dim=0)
            
            # 层归一化
            h = self.layer_norm(h)
            
            # 应用中间层处理
            h = self.mid_layers[i](h)
            
            # 应用注意力 (如果启用)
            if self.use_attention:
                attn = self.attention(h)
                h = h * attn
        
        # 输出投影
        out = self.output_proj(h)
        out = self.dropout(out)
        
        # 残差连接
        return identity + out


class MultiscaleKANBlock(nn.Module):
    """
    多尺度KAN模块，处理不同空间尺度的特征
    """
    def __init__(self, in_channels, hidden_dim=128, scales=[1, 2, 4], layers=3, dropout=0.2):
        super(MultiscaleKANBlock, self).__init__()
        self.scales = scales
        
        # 为每个尺度创建KAN模块
        self.kan_blocks = nn.ModuleList([
            EnhancedKANBlock(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                heads=4,
                layers=layers,
                dropout=dropout,
                activation='mish',
                use_attention=True
            ) for _ in scales
        ])
        
        # 缩放和融合模块
        self.scale_fuse = nn.Conv2d(in_channels * len(scales), in_channels, kernel_size=1)
    
    def forward(self, x):
        outputs = []
        
        # 对每个尺度应用KAN处理
        for i, scale in enumerate(self.scales):
            # 缩小特征图 (如果尺度>1)
            if scale > 1:
                # 下采样
                scaled_x = F.avg_pool2d(x, kernel_size=scale, stride=scale)
                
                # 应用KAN模块
                scaled_out = self.kan_blocks[i](scaled_x)
                
                # 上采样回原始尺寸
                out = F.interpolate(scaled_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            else:
                # 直接应用KAN模块 (原始尺度)
                out = self.kan_blocks[i](x)
            
            outputs.append(out)
        
        # 合并不同尺度的输出
        merged = torch.cat(outputs, dim=1)
        result = self.scale_fuse(merged)
        
        # 残差连接
        return x + result


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 64, 32, 32).to(device)
    
    # 测试增强型KAN模块
    kan_block = EnhancedKANBlock(
        in_channels=64,
        hidden_dim=128,
        heads=4,
        layers=5,
        dropout=0.2,
        activation='gelu',
        use_attention=True
    ).to(device)
    
    y = kan_block(x)
    print(f"输入形状: {x.shape}, 输出形状: {y.shape}")
    
    # 测试多尺度KAN模块
    multiscale_kan = MultiscaleKANBlock(
        in_channels=64,
        hidden_dim=128,
        scales=[1, 2, 4],
        layers=3,
        dropout=0.2
    ).to(device)
    
    z = multiscale_kan(x)
    print(f"多尺度输入形状: {x.shape}, 输出形状: {z.shape}") 