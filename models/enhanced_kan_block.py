#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnhancedKANBlock(nn.Module):
    """增强版KAN块，使用门控单元和注意力机制"""
    def __init__(self, in_channels, hidden_dim=256, heads=4, layers=5, dropout=0.3, 
                 activation='gelu', use_attention=True):
        super(EnhancedKANBlock, self).__init__()
        
        self.use_attention = use_attention
        self.layers = layers
        
        # 映射输入通道到隐藏维度
        self.input_projection = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # 构建KAN层
        self.kan_layers = nn.ModuleList()
        
        for _ in range(layers):
            # 卷积层
            conv_block = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                self._get_activation(activation),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim)
            )
            
            # 门控单元
            gate = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.Sigmoid()
            )
            
            # 注意力机制
            if use_attention:
                attn = nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=heads,
                    dropout=dropout
                )
                
                self.kan_layers.append(nn.ModuleDict({
                    'conv': conv_block,
                    'gate': gate,
                    'attn': attn,
                    'norm': nn.LayerNorm([hidden_dim]),
                    'dropout': nn.Dropout(dropout)
                }))
            else:
                self.kan_layers.append(nn.ModuleDict({
                    'conv': conv_block,
                    'gate': gate,
                    'dropout': nn.Dropout(dropout)
                }))
        
        # 输出映射
        self.output_projection = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
        # 最终激活层
        self.final_activation = self._get_activation(activation)
    
    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'leaky_relu':
            return nn.LeakyReLU(0.2, inplace=True)
        else:
            return nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 输入映射
        h = self.input_projection(x)
        
        # 应用KAN层
        for i in range(self.layers):
            layer = self.kan_layers[i]
            
            # 残差连接准备
            residual = h
            
            # 应用卷积块
            conv_out = layer['conv'](h)
            
            # 应用门控
            gate_out = layer['gate'](h)
            gated_out = conv_out * gate_out
            
            # 添加残差连接
            h = residual + gated_out
            
            # 应用注意力（如果启用）
            if self.use_attention:
                # 为注意力调整维度
                b, c, height, width = h.shape
                flat_h = h.flatten(2).permute(2, 0, 1)  # (seq_len, batch, hidden_dim)
                
                # 应用多头注意力
                attn_out, _ = layer['attn'](flat_h, flat_h, flat_h)
                
                # 应用层归一化和dropout
                attn_out = layer['norm'](attn_out)
                attn_out = layer['dropout'](attn_out)
                
                # 调整回原始维度
                attn_out = attn_out.permute(1, 2, 0).view(b, c, height, width)
                
                # 添加残差连接
                h = h + attn_out
            else:
                # 仅应用dropout
                h = layer['dropout'](h)
        
        # 输出映射
        out = self.output_projection(h)
        
        # 应用残差连接并激活
        out = self.final_activation(out + x)
        
        return out


class MultiscaleKANBlock(nn.Module):
    """多尺度KAN块，处理不同尺度下的特征"""
    def __init__(self, in_channels, hidden_dim=256, scales=[1, 2, 4], layers=3, dropout=0.3):
        super(MultiscaleKANBlock, self).__init__()
        
        self.scales = scales
        
        # 不同尺度的KAN块
        self.kan_branches = nn.ModuleList()
        
        for scale in scales:
            # 适当的池化层
            if scale > 1:
                pool = nn.AvgPool2d(kernel_size=scale, stride=scale)
            else:
                pool = nn.Identity()
            
            # 对应尺度的KAN块
            kan = EnhancedKANBlock(
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                layers=layers,
                dropout=dropout,
                use_attention=(scale <= 2)  # 对较小尺度使用注意力
            )
            
            self.kan_branches.append(nn.ModuleDict({
                'pool': pool,
                'kan': kan
            }))
        
        # 特征融合
        self.fusion = nn.Conv2d(in_channels * len(scales), in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        outputs = []
        
        # 应用不同尺度的KAN块
        for i, scale in enumerate(self.scales):
            branch = self.kan_branches[i]
            
            # 下采样
            down = branch['pool'](x)
            
            # 应用KAN
            kan_out = branch['kan'](down)
            
            # 上采样回原始尺寸
            if scale > 1:
                kan_out = F.interpolate(kan_out, size=x.shape[2:], mode='bilinear', align_corners=False)
            
            outputs.append(kan_out)
        
        # 通道维度上连接所有输出
        concat_out = torch.cat(outputs, dim=1)
        
        # 融合特征
        out = self.relu(self.bn(self.fusion(concat_out)))
        
        return out


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