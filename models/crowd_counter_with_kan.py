#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# 尝试导入KAN
try:
    from efficient_kan import KAN
    HAS_KAN = True
except ImportError:
    print("警告: 找不到efficient_kan模块，将使用替代实现")
    HAS_KAN = False
    # 定义替代的KAN实现
    class KAN(nn.Module):
        def __init__(self, in_dim, hid_dim=64, depth=1, dropout=0.1):
            super().__init__()
            self.in_dim = in_dim
            self.hid_dim = hid_dim
            self.depth = depth
            
            # 使用MLP替代KAN
            layers = [
                nn.Linear(in_dim, hid_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            for _ in range(depth-1):
                layers.extend([
                    nn.Linear(hid_dim, hid_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
            
            self.network = nn.Sequential(*layers)
            self.out_proj = nn.Linear(hid_dim, in_dim)
        
        def forward(self, x):
            # 保存原始形状
            orig_shape = x.shape
            # 如果输入是多维张量，将其展平为(batch_size, -1)
            if len(orig_shape) > 2:
                x = x.view(orig_shape[0], -1)
            
            x = self.network(x)
            x = self.out_proj(x)
            
            # 恢复原始形状
            if len(orig_shape) > 2:
                x = x.view(orig_shape)
            
            return x

class KANBlock(nn.Module):
    """Kolmogorov-Arnold Network (KAN) 增强模块"""
    def __init__(self, in_channels, hidden_dim=64, layers=3):
        super(KANBlock, self).__init__()
        self.layers = layers
        
        # 定义可学习的基函数参数
        self.alpha = nn.Parameter(torch.randn(hidden_dim))
        self.beta = nn.Parameter(torch.randn(hidden_dim))
        self.gamma = nn.Parameter(torch.randn(hidden_dim))
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # 中间层
        self.mid_layers = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1) 
            for _ in range(layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
    def forward(self, x):
        # 输入投影
        h = self.input_proj(x)
        
        # 应用KAN基函数和中间层
        for i in range(self.layers):
            # 应用基函数: alpha * sin(beta * x + gamma)
            h_base = self.alpha.view(1, -1, 1, 1) * torch.sin(
                self.beta.view(1, -1, 1, 1) * h + self.gamma.view(1, -1, 1, 1)
            )
            
            # 应用卷积层
            h = self.mid_layers[i](h_base)
            h = F.relu(h)
        
        # 输出投影
        out = self.output_proj(h)
        
        # 残差连接
        return x + out

class MobileNetBackbone(nn.Module):
    """使用MobileNetV3作为特征提取器的主干网络，保留空间维度"""
    def __init__(self, pretrained=True):
        super(MobileNetBackbone, self).__init__()
        
        # 加载预训练的MobileNetV3模型
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mobilenet = mobilenet_v3_small(weights=weights)
        
        # 去掉最后的全局池化和分类器层，只保留特征提取部分
        # MobileNetV3结构：features -> avgpool -> flatten -> classifier
        self.features = mobilenet.features
        
        # 输出通道数
        self.out_channels = 576  # MobileNetV3-Small最后一层特征的通道数
    
    def forward(self, x):
        """前向传播
        Args:
            x: 输入图像 [B, 3, H, W]
        Returns:
            features: 提取的特征 [B, C, H', W']，保留空间维度
        """
        # 通过MobileNetV3的特征提取部分
        features = self.features(x)
        return features

class CrowdCounterWithKAN(nn.Module):
    """使用KAN模块增强的人群计数器 - 超轻量版本，确保非负密度输出"""
    def __init__(self, backbone='mobilenet', model_type='basic', num_classes=1, pretrained=True, 
                 kan_layers=5, kan_hidden_dim=64, kan_heads=4, activation='relu',
                 use_fpn=False, use_multiscale_kan=False, dropout=0.3):
        """
        初始化人群计数模型
        Args:
            backbone: 使用的主干网络，'efficientnet'或'mobilenet'或'resnet50'
            model_type: 模型类型，'basic', 'dual_path', 或其他
            num_classes: 输出类别数量
            pretrained: 是否使用预训练权重
            kan_layers: KAN层数
            kan_hidden_dim: KAN隐藏层维度
            kan_heads: KAN多头注意力的头数
            activation: 激活函数类型
            use_fpn: 是否使用特征金字塔
            use_multiscale_kan: 是否在多尺度上应用KAN
            dropout: dropout比率
        """
        super(CrowdCounterWithKAN, self).__init__()
        self.backbone_name = backbone
        self.model_type = model_type
        self.num_classes = num_classes
        self.kan_layers = kan_layers
        self.kan_hidden_dim = kan_hidden_dim
        self.kan_heads = kan_heads
        self.activation = activation
        self.use_fpn = use_fpn
        self.use_multiscale_kan = use_multiscale_kan
        self.dropout = dropout
        
        # 使用简单的自定义MobileNet骨干网络
        self.backbone = MobileNetBackbone(pretrained=pretrained)
        feature_dim = self.backbone.out_channels  # 576 for MobileNetV3 Small
        
        # 轻量级KAN增强模块
        self.kan_block = nn.Sequential(
            nn.Conv2d(feature_dim, 128, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, feature_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 非常简化的上采样架构 - 直接从特征到密度图
        self.conv1 = nn.Conv2d(feature_dim, 128, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # 最终密度图预测
        self.density_pred = nn.Conv2d(32, 1, kernel_size=1)
        
    def forward(self, x):
        """前向传播
        Args:
            x: 输入图像，形状 [batch_size, 3, height, width]
        Returns:
            density_map: 密度图，形状 [batch_size, 1, output_height, output_width]
        """
        # 记录输入尺寸以便最终调整
        input_size = (x.shape[2], x.shape[3])
        
        # 特征提取 - 现在确保输出是 [B, C, H', W']
        features = self.backbone(x)
        
        # 轻量级特征增强
        attn = self.kan_block(features)
        enhanced_features = features * attn
        
        # 简化的特征处理 - 不使用转置卷积
        x = self.conv1(enhanced_features)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # 第一次上采样
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # 第二次上采样
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv3(x)
        x = F.relu(x)
        
        # 密度图预测
        x = self.density_pred(x)
        
        # 最终调整到输入尺寸
        density_map = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        # 确保密度图预测值非负（使用ReLU）
        density_map = F.relu(density_map)
        
        return density_map

class DualPathCrowdCounter(nn.Module):
    """双路径人群计数器，处理RGB图像和质量信息"""
    def __init__(self, pretrained=True):
        super(DualPathCrowdCounter, self).__init__()
        
        # RGB分支
        self.rgb_backbone = MobileNetBackbone(pretrained=pretrained)
        
        # 质量分支
        self.quality_backbone = MobileNetBackbone(pretrained=pretrained)
        
        # 特征融合
        self.fusion = nn.Conv2d(
            self.rgb_backbone.out_channels + self.quality_backbone.out_channels, 
            512, 
            kernel_size=1
        )
        
        # KAN增强模块
        self.kan_block = KANBlock(in_channels=512, hidden_dim=256, layers=3)
        
        # 上采样和密度图生成
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.upconv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        
        # 最终密度图预测
        self.predict = nn.Conv2d(16, 1, kernel_size=1)
    
    def forward(self, rgb_input, quality_input=None):
        # 记录输入尺寸以便最终调整
        input_size = (rgb_input.shape[2], rgb_input.shape[3])
        
        # 如果没有提供质量输入，使用RGB输入替代
        if quality_input is None:
            quality_input = rgb_input
        
        # 特征提取
        rgb_features = self.rgb_backbone(rgb_input)
        quality_features = self.quality_backbone(quality_input)
        
        # 特征融合
        fused_features = torch.cat([rgb_features, quality_features], dim=1)
        fused_features = self.fusion(fused_features)
        
        # 应用KAN增强
        enhanced_features = self.kan_block(fused_features)
        
        # 上采样和密度图生成
        x = F.relu(self.upconv1(enhanced_features))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        x = F.relu(self.upconv5(x))
        
        # 密度图预测
        density_map = self.predict(x)
        
        # 应用softplus确保非负输出
        density_map = F.softplus(density_map, beta=10)
        
        # 确保输出尺寸与输入相同
        if density_map.shape[2:] != input_size:
            density_map = F.interpolate(density_map, size=input_size, mode='bilinear', align_corners=False)
        
        return density_map

class MobileNetMergedCrowdCounter(nn.Module):
    """特征合并的人群计数器，使用两个MobileNet分支"""
    def __init__(self, pretrained=True, rgb_weights_path=None, quality_weights_path=None):
        super(MobileNetMergedCrowdCounter, self).__init__()
        
        # RGB分支
        self.rgb_backbone = MobileNetBackbone(pretrained=pretrained)
        
        # 质量分支
        self.quality_backbone = MobileNetBackbone(pretrained=pretrained)
        
        # 加载预训练权重
        if rgb_weights_path and os.path.exists(rgb_weights_path):
            self.rgb_backbone.load_state_dict(torch.load(rgb_weights_path))
            print(f"已加载RGB分支权重: {rgb_weights_path}")
        
        if quality_weights_path and os.path.exists(quality_weights_path):
            self.quality_backbone.load_state_dict(torch.load(quality_weights_path))
            print(f"已加载质量分支权重: {quality_weights_path}")
        
        # 特征融合通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.rgb_backbone.out_channels, self.rgb_backbone.out_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.rgb_backbone.out_channels // 8, self.rgb_backbone.out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 合并后的特征处理
        self.process = nn.Conv2d(
            self.rgb_backbone.out_channels, 
            self.rgb_backbone.out_channels,
            kernel_size=3,
            padding=1
        )
        
        # KAN增强模块
        self.kan_block = KANBlock(
            in_channels=self.rgb_backbone.out_channels, 
            hidden_dim=128, 
            layers=3
        )
        
        # 上采样和密度图生成
        self.upconv1 = nn.ConvTranspose2d(self.rgb_backbone.out_channels, 256, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.upconv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        
        # 最终密度图预测
        self.predict = nn.Conv2d(16, 1, kernel_size=1)
    
    def forward(self, rgb_input, quality_input=None):
        # 记录输入尺寸以便最终调整
        input_size = (rgb_input.shape[2], rgb_input.shape[3])
        
        # 如果没有提供质量输入，使用RGB输入替代
        if quality_input is None:
            quality_input = rgb_input
        
        # 特征提取
        rgb_features = self.rgb_backbone(rgb_input)
        quality_features = self.quality_backbone(quality_input)
        
        # 通道注意力
        attn = self.channel_attn(quality_features)
        
        # 特征融合: RGB特征 + 质量加权特征
        merged_features = rgb_features + quality_features * attn
        
        # 合并特征处理
        merged_features = self.process(merged_features)
        
        # 应用KAN增强
        enhanced_features = self.kan_block(merged_features)
        
        # 上采样和密度图生成
        x = F.relu(self.upconv1(enhanced_features))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        x = F.relu(self.upconv5(x))
        
        # 密度图预测
        density_map = self.predict(x)
        
        # 应用softplus确保非负输出
        density_map = F.softplus(density_map, beta=10)
        
        # 确保输出尺寸与输入相同
        if density_map.shape[2:] != input_size:
            density_map = F.interpolate(density_map, size=input_size, mode='bilinear', align_corners=False)
        
        return density_map 