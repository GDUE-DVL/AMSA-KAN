#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# 尝试导入KAN模块
try:
    from enhanced_kan_block import EnhancedKANBlock, MultiscaleKANBlock
except ImportError:
    print("警告: 找不到enhanced_kan_block模块，使用相对导入")
    from enhanced_kan_block import EnhancedKANBlock, MultiscaleKANBlock

class EfficientNetBackbone(nn.Module):
    """使用EfficientNet作为特征提取器的主干网络"""
    def __init__(self, pretrained=True):
        super(EfficientNetBackbone, self).__init__()
        
        # 加载预训练的EfficientNet模型
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.efficientnet = efficientnet_b2(weights=weights)
        
        # 移除分类头
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-1])
        
        # 获取特征维度 (EfficientNet B2的特征维度为1408)
        self.out_channels = 1408
    
    def forward(self, x):
        features = self.efficientnet(x)
        return features

class MobileNetV3Backbone(nn.Module):
    """使用MobileNetV3作为特征提取器的主干网络"""
    def __init__(self, pretrained=True):
        super(MobileNetV3Backbone, self).__init__()
        
        # 加载预训练的MobileNetV3模型
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        self.mobilenet = mobilenet_v3_large(weights=weights)
        
        # 移除分类头
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])
        
        # 获取特征维度
        self.out_channels = 960  # MobileNetV3-Large的最后一层特征通道数
    
    def forward(self, x):
        features = self.mobilenet(x)
        return features

class FeaturePyramidNetwork(nn.Module):
    """特征金字塔网络，处理多尺度特征"""
    def __init__(self, in_channels, out_channels=256):
        super(FeaturePyramidNetwork, self).__init__()
        
        # 自适应池化以生成多尺度特征
        self.p1 = nn.AdaptiveAvgPool2d(1)    # 全局特征
        self.p2 = nn.AdaptiveAvgPool2d(2)    # 2x2特征
        self.p3 = nn.AdaptiveAvgPool2d(4)    # 4x4特征
        self.p4 = nn.AdaptiveAvgPool2d(8)    # 8x8特征
        
        # 1x1卷积调整通道
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # 融合多尺度特征
        self.fusion = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 生成多尺度特征
        f1 = self.conv1(self.p1(x))
        f2 = self.conv2(self.p2(x))
        f3 = self.conv3(self.p3(x))
        f4 = self.conv4(self.p4(x))
        
        # 上采样到相同尺寸
        target_size = f4.size()[2:]
        f1 = F.interpolate(f1, size=target_size, mode='bilinear', align_corners=False)
        f2 = F.interpolate(f2, size=target_size, mode='bilinear', align_corners=False)
        f3 = F.interpolate(f3, size=target_size, mode='bilinear', align_corners=False)
        
        # 特征融合
        fusion = torch.cat([f1, f2, f3, f4], dim=1)
        out = self.relu(self.bn(self.fusion(fusion)))
        
        return out

class AdvancedCrowdCounterWithKAN(nn.Module):
    """高级KAN人群计数器，使用增强型KAN模块和多种正则化技术"""
    def __init__(self, backbone='efficientnet', pretrained=True, dropout=0.3, 
                 kan_layers=5, kan_hidden_dim=256, kan_heads=4, activation='gelu'):
        super(AdvancedCrowdCounterWithKAN, self).__init__()
        
        # 选择骨干网络
        if backbone == 'efficientnet':
            self.backbone = EfficientNetBackbone(pretrained=pretrained)
        else:
            self.backbone = MobileNetV3Backbone(pretrained=pretrained)
        
        # 特征金字塔网络
        self.fpn = FeaturePyramidNetwork(self.backbone.out_channels, out_channels=512)
        
        # 增强型KAN模块
        self.kan_block = EnhancedKANBlock(
            in_channels=512,
            hidden_dim=kan_hidden_dim,
            heads=kan_heads,
            layers=kan_layers,
            dropout=dropout,
            activation=activation,
            use_attention=True
        )
        
        # 多尺度KAN模块
        self.multiscale_kan = MultiscaleKANBlock(
            in_channels=512,
            hidden_dim=kan_hidden_dim,
            scales=[1, 2, 4],
            layers=3,
            dropout=dropout
        )
        
        # 多分辨率解码器
        self.decoder = nn.ModuleList([
            # 第一级解码器
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ),
            # 第二级解码器
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ),
            # 第三级解码器
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ),
            # 第四级解码器
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 可学习的缩放因子
        self.scale_factor = nn.Parameter(torch.tensor([0.01]), requires_grad=True)
        
        # 最终密度图预测
        self.predict = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x):
        # 记录输入尺寸以便最终调整
        input_size = (x.shape[2], x.shape[3])
        
        # 特征提取
        features = self.backbone(x)
        
        # 应用特征金字塔网络
        fpn_features = self.fpn(features)
        
        # 应用增强型KAN模块
        kan_features = self.kan_block(fpn_features)
        
        # 应用多尺度KAN模块
        multiscale_features = self.multiscale_kan(kan_features)
        
        # 应用多分辨率解码器
        decoder_out = multiscale_features
        for decoder_layer in self.decoder:
            decoder_out = decoder_layer(decoder_out)
        
        # 密度图预测
        density_map = self.predict(decoder_out)
        
        # 应用缩放因子
        density_map = density_map * torch.abs(self.scale_factor)
        
        # 确保非负输出
        density_map = F.softplus(density_map, beta=10)
        
        # 最终调整到输入尺寸
        density_map = F.interpolate(density_map, size=input_size, mode='bilinear', align_corners=False)
        
        # 确保输出合理
        density_map = torch.clamp(density_map, min=0.0, max=1000.0)
        
        return density_map

class DualStreamCrowdCounter(nn.Module):
    """双流人群计数模型，处理RGB和质量图"""
    def __init__(self, pretrained=True, dropout=0.3, 
                 kan_layers=5, kan_hidden_dim=256, fusion_type='attention'):
        super(DualStreamCrowdCounter, self).__init__()
        
        # RGB分支
        self.rgb_backbone = EfficientNetBackbone(pretrained=pretrained)
        
        # 质量分支
        self.quality_backbone = MobileNetV3Backbone(pretrained=pretrained)
        
        # 特征融合模块
        self.fusion_type = fusion_type
        if fusion_type == 'attention':
            # 通道和空间注意力融合
            self.channel_attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.rgb_backbone.out_channels, self.rgb_backbone.out_channels // 8, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.rgb_backbone.out_channels // 8, self.rgb_backbone.out_channels, kernel_size=1),
                nn.Sigmoid()
            )
            
            self.spatial_attn = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3),
                nn.Sigmoid()
            )
            
            self.fusion_conv = nn.Conv2d(
                self.rgb_backbone.out_channels + self.quality_backbone.out_channels,
                512,
                kernel_size=1
            )
        else:
            # 简单连接融合
            self.fusion_conv = nn.Conv2d(
                self.rgb_backbone.out_channels + self.quality_backbone.out_channels,
                512,
                kernel_size=1
            )
        
        # 增强KAN处理
        self.kan_block = EnhancedKANBlock(
            in_channels=512,
            hidden_dim=kan_hidden_dim,
            heads=4,
            layers=kan_layers,
            dropout=dropout,
            activation='gelu',
            use_attention=True
        )
        
        # 解码器模块
        self.decoder = nn.ModuleList([
            # 第一级解码器
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ),
            # 第二级解码器
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ),
            # 第三级解码器
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ),
            # 第四级解码器
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 可学习的缩放因子
        self.scale_factor = nn.Parameter(torch.tensor([0.01]), requires_grad=True)
        
        # 最终密度图预测
        self.predict = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, rgb_input, quality_input=None):
        # 记录输入尺寸以便最终调整
        input_size = (rgb_input.shape[2], rgb_input.shape[3])
        
        # 如果未提供质量输入，使用RGB代替
        if quality_input is None:
            quality_input = rgb_input
        
        # 特征提取
        rgb_features = self.rgb_backbone(rgb_input)
        quality_features = self.quality_backbone(quality_input)
        
        # 特征融合
        if self.fusion_type == 'attention':
            # 通道注意力
            channel_attn = self.channel_attn(quality_features)
            enhanced_rgb = rgb_features * channel_attn
            
            # 空间注意力
            spatial_avg = torch.mean(enhanced_rgb, dim=1, keepdim=True)
            spatial_max, _ = torch.max(enhanced_rgb, dim=1, keepdim=True)
            spatial_concat = torch.cat([spatial_avg, spatial_max], dim=1)
            spatial_attn = self.spatial_attn(spatial_concat)
            
            # 应用空间注意力
            enhanced_rgb = enhanced_rgb * spatial_attn
            
            # 连接特征
            fused_features = torch.cat([enhanced_rgb, quality_features], dim=1)
        else:
            # 直接连接
            fused_features = torch.cat([rgb_features, quality_features], dim=1)
        
        # 调整通道数
        fused_features = self.fusion_conv(fused_features)
        
        # KAN增强
        enhanced_features = self.kan_block(fused_features)
        
        # 应用解码器
        decoder_out = enhanced_features
        for decoder_layer in self.decoder:
            decoder_out = decoder_layer(decoder_out)
        
        # 密度图预测
        density_map = self.predict(decoder_out)
        
        # 应用缩放因子
        density_map = density_map * torch.abs(self.scale_factor)
        
        # 确保非负输出
        density_map = F.softplus(density_map, beta=10)
        
        # 最终调整到输入尺寸
        density_map = F.interpolate(density_map, size=input_size, mode='bilinear', align_corners=False)
        
        # 确保输出合理
        density_map = torch.clamp(density_map, min=0.0, max=1000.0)
        
        return density_map

def get_crowd_counter_model(model_type='advanced', **kwargs):
    """
    获取人群计数模型
    
    参数:
        model_type: 模型类型 ('advanced', 'dual_stream')
        **kwargs: 其他模型参数
    
    返回:
        model: 人群计数模型
    """
    if model_type == 'advanced':
        return AdvancedCrowdCounterWithKAN(**kwargs)
    elif model_type == 'dual_stream':
        return DualStreamCrowdCounter(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 384, 384).to(device)
    
    # 测试高级人群计数模型
    model = AdvancedCrowdCounterWithKAN().to(device)
    y = model(x)
    print(f"高级模型 - 输入形状: {x.shape}, 输出形状: {y.shape}")
    
    # 测试双流人群计数模型
    dual_model = DualStreamCrowdCounter().to(device)
    z = dual_model(x, x)
    print(f"双流模型 - 输入形状: {x.shape}, 输出形状: {z.shape}")
    
    # 打印模型参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"高级模型参数量: {count_parameters(model) / 1e6:.2f}M")
    print(f"双流模型参数量: {count_parameters(dual_model) / 1e6:.2f}M") 