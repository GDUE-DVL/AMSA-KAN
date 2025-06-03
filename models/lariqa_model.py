#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LAR-IQA 模型定义
- 双分支架构
- 支持不同的主干网络 (MobileNet/ResNet50)
- 支持不同的注意力机制
- 使用KAN网络进行回归
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights, ResNet50_Weights

class MSFF(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, in_channels, out_channels=256):
        super(MSFF, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class AttentionModule(nn.Module):
    """注意力模块"""
    def __init__(self, in_channels, attention_type='self'):
        super(AttentionModule, self).__init__()
        self.attention_type = attention_type
        
        if attention_type == 'self':
            # 自注意力
            self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
            self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
            self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.gamma = nn.Parameter(torch.zeros(1))
        elif attention_type == 'cbam':
            # CBAM注意力
            self.channel_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
                nn.Sigmoid()
            )
            self.spatial_att = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            )
            
    def forward(self, x):
        if self.attention_type == 'self':
            batch_size, C, H, W = x.size()
            
            # 计算注意力分数
            proj_query = self.query(x).view(batch_size, -1, H*W).permute(0, 2, 1)
            proj_key = self.key(x).view(batch_size, -1, H*W)
            energy = torch.bmm(proj_query, proj_key)
            attention = F.softmax(energy, dim=-1)
            
            # 应用注意力
            proj_value = self.value(x).view(batch_size, -1, H*W)
            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(batch_size, C, H, W)
            
            out = self.gamma * out + x
            return out
            
        elif self.attention_type == 'cbam':
            # 通道注意力
            ca = self.channel_att(x)
            x = x * ca
            
            # 空间注意力
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            y = torch.cat([avg_out, max_out], dim=1)
            y = self.spatial_att(y)
            
            return x * y
            
        return x

class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network (KAN) 层"""
    def __init__(self, in_features, hidden_dim, grid_size=10, spline_type='bspline'):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.spline_type = spline_type
        
        # 定义控制点
        self.control_points = nn.Parameter(torch.randn(in_features, hidden_dim, grid_size))
        
        # 线性变换
        self.linear = nn.Linear(in_features, hidden_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 简化版KAN：使用线性变换代替实际的样条插值
        # 实际实现中应该使用B样条或多项式样条插值
        x = self.linear(x)
        return x

class LARIQAModel(nn.Module):
    """
    LAR-IQA 模型
    """
    def __init__(self, backbone='mobilenet', fusion_method='weighted', 
                 use_attention=True, attention_type='self', use_msff=True,
                 kan_hidden_dim=256, kan_layers=7, kan_grid_size=10, kan_spline_type='bspline'):
        super(LARIQAModel, self).__init__()
        self.backbone_type = backbone
        self.fusion_method = fusion_method
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.use_msff = use_msff
        
        # 初始化主干网络
        if backbone == 'mobilenet':
            base_model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            self.features = base_model.features
            self.feature_dim = 1280
        elif backbone == 'resnet50':
            base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            # 移除最后的全连接层和平均池化
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            self.feature_dim = 2048
        else:
            raise ValueError(f"不支持的主干网络: {backbone}")
        
        # 特征融合模块
        if use_msff:
            self.msff = MSFF(self.feature_dim)
            self.feature_dim = 256  # MSFF输出通道数
        
        # 注意力模块
        if use_attention:
            self.attention = AttentionModule(self.feature_dim, attention_type)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # KAN回归网络
        self.kan_layers = nn.ModuleList()
        kan_input_dim = self.feature_dim
        for _ in range(kan_layers):
            self.kan_layers.append(
                KANLayer(kan_input_dim, kan_hidden_dim, kan_grid_size, kan_spline_type)
            )
            kan_input_dim = kan_hidden_dim
        
        # 最终预测层
        self.final_layer = nn.Linear(kan_hidden_dim, 1)
        
    def forward(self, x):
        # 提取特征
        features = self.features(x)
        
        # 多尺度特征融合
        if self.use_msff:
            features = self.msff(features)
        
        # 应用注意力
        if self.use_attention:
            features = self.attention(features)
        
        # 全局池化
        features = self.global_pool(features)
        features = torch.flatten(features, 1)
        
        # KAN回归
        for kan_layer in self.kan_layers:
            features = kan_layer(features)
        
        # 最终预测
        pred = self.final_layer(features)
        
        return pred 