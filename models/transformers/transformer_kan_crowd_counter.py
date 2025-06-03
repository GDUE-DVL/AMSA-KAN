#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TransformerKANCrowdCounter模型 - 用于高精度人群计数
结合了Transformer和KAN（Kolmogorov-Arnold Network）的优势，
专为UCF-QNRF等高变化范围人群密度数据集设计。
"""

import os
import sys
import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# 尝试导入KAN
try:
    from efficient_kan import KAN
    HAS_KAN = True
except ImportError:
    print("警告: 找不到efficient_kan模块，将使用替代实现")
    HAS_KAN = False

# AdaptiveKANModule - 自适应KAN模块
class AdaptiveKANModule(nn.Module):
    """自适应KAN模块 - 动态调整核心参数"""
    def __init__(self, in_channels, hidden_dim=256, depth=8, heads=8):
        super(AdaptiveKANModule, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.heads = heads
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, hidden_dim, 64, 64))
        
        # KAN层或替代方案
        if HAS_KAN:
            self.kan_layers = nn.ModuleList([
                KAN([hidden_dim, hidden_dim]) 
                for _ in range(depth)
            ])
        else:
            # 使用自注意力 + FFN作为替代
            self.attention_layers = nn.ModuleList([
                nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
                for _ in range(depth)
            ])
            self.ffn_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                )
                for _ in range(depth)
            ])
            self.layer_norms_1 = nn.ModuleList([
                nn.LayerNorm(hidden_dim) 
                for _ in range(depth)
            ])
            self.layer_norms_2 = nn.ModuleList([
                nn.LayerNorm(hidden_dim) 
                for _ in range(depth)
            ])
        
        # 输出投影
        self.output_proj = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, _, h, w = x.shape
        
        # 输入投影
        x_proj = self.input_proj(x)
        
        # 添加位置编码
        pos_embed = F.interpolate(self.pos_embedding, size=(h, w), mode='bilinear', align_corners=False)
        x_pos = x_proj + pos_embed
        
        if HAS_KAN:
            # 使用KAN处理（直接在空间维度上操作）
            x_kan = x_pos
            for kan_layer in self.kan_layers:
                # 重塑张量以适应KAN输入
                b, c, h, w = x_kan.shape
                x_reshaped = x_kan.permute(0, 2, 3, 1).reshape(-1, c)
                
                # 应用KAN
                x_processed = kan_layer(x_reshaped)
                
                # 重塑回原始形状
                x_kan = x_processed.reshape(b, h, w, c).permute(0, 3, 1, 2)
                
                # 残差连接
                x_kan = x_kan + x_pos
        else:
            # 使用自注意力处理
            # 重塑为序列 [B, C, H, W] -> [B, HW, C]
            x_seq = x_pos.flatten(2).permute(0, 2, 1)
            
            # 应用多层注意力
            x_attn = x_seq
            for i in range(self.depth):
                # 第一个子层：多头自注意力
                residual = x_attn
                x_attn = self.layer_norms_1[i](x_attn)
                x_attn, _ = self.attention_layers[i](x_attn, x_attn, x_attn)
                x_attn = residual + x_attn
                
                # 第二个子层：前馈网络
                residual = x_attn
                x_attn = self.layer_norms_2[i](x_attn)
                x_attn = self.ffn_layers[i](x_attn)
                x_attn = residual + x_attn
            
            # 重塑回空间表示 [B, HW, C] -> [B, C, H, W]
            x_kan = x_attn.permute(0, 2, 1).reshape(batch_size, -1, h, w)
        
        # 输出投影
        out = self.output_proj(x_kan)
        
        # 残差连接
        return out + x

# DensityPredictionHead - 密度图预测头
class DensityPredictionHead(nn.Module):
    """改进的密度图预测头，使用深度可分离卷积和注意力机制"""
    def __init__(self, in_channels, scale_factor=4):
        super(DensityPredictionHead, self).__init__()
        self.scale_factor = scale_factor
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        # 深度可分离卷积模块 - 降低参数量同时保持性能
        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels//2, kernel_size=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        
        # 密度估计分支
        self.density_branch = nn.Sequential(
            nn.Conv2d(in_channels//2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        
    def forward(self, x):
        # 应用通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 应用空间注意力
        sa = self.spatial_attention(x)
        x = x * sa
        
        # 深度可分离卷积
        x = self.dwconv(x)
        
        # 上采样到原始分辨率
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        
        # 生成密度图
        density = self.density_branch(x)
        
        # 应用ReLU确保预测值非负
        density = F.relu(density)
        
        return density

# GlobalCountHead - 全局计数头
class GlobalCountHead(nn.Module):
    """全局计数头，直接从特征预测总计数"""
    def __init__(self, in_channels):
        super(GlobalCountHead, self).__init__()
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 计数预测器
        self.count_predictor = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 预测计数值（确保非负）
        count = self.count_predictor(x)
        count = F.relu(count)
        
        return count

# MultiScaleProcessor - 多尺度处理器
class MultiScaleProcessor(nn.Module):
    """多尺度处理器，根据场景特点自适应调整密度图"""
    def __init__(self):
        super(MultiScaleProcessor, self).__init__()
        
        # 密度图调整参数
        self.density_scale = nn.Parameter(torch.tensor(1.0))
        self.density_shift = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, density_map, count=None):
        # 应用密度图缩放和平移
        density_map = density_map * self.density_scale + self.density_shift
        
        # 确保密度图非负
        density_map = F.relu(density_map)
        
        # 如果有全局计数预测，确保密度图与之一致
        if count is not None:
            batch_size = density_map.size(0)
            
            # 计算当前密度图的总和
            current_sum = density_map.sum(dim=(1, 2, 3))
            
            # 避免除零
            current_sum = torch.clamp(current_sum, min=1e-6)
            
            # 计算每个样本的缩放因子
            scale_factor = count.squeeze() / current_sum
            
            # 应用缩放因子（保持空间分布）
            scaled_density = density_map * scale_factor.view(batch_size, 1, 1, 1)
            
            return scaled_density
        
        return density_map

# TransformerKANCrowdCounter - 主模型类
class TransformerKANCrowdCounter(nn.Module):
    """结合Transformer和KAN的混合架构人群计数模型"""
    def __init__(self, backbone='swin_base_patch4_window7_224', pretrained=True, 
                 use_kan=True, density_output_size=(768, 768)):
        super(TransformerKANCrowdCounter, self).__init__()
        
        self.density_output_size = density_output_size
        
        # 使用Swin Transformer作为主干网络
        self.backbone = timm.create_model(backbone, 
                                         pretrained=pretrained, 
                                         features_only=True)
        
        # 获取主干网络的输出通道数（因模型而异）
        # 假设为Swin-Base的输出通道
        self.feature_channels = 512
        
        # 自适应KAN增强模块
        self.kan_module = AdaptiveKANModule(
            in_channels=self.feature_channels,
            hidden_dim=256,
            depth=8,
            heads=8
        )
        
        # 密度预测器
        self.density_head = DensityPredictionHead(
            in_channels=self.feature_channels,
            scale_factor=4
        )
        
        # 计数预测分支
        self.count_head = GlobalCountHead(self.feature_channels)
        
        # 多分辨率处理模块
        self.multi_scale_processor = MultiScaleProcessor()
    
    def forward(self, x):
        # 保存输入尺寸
        input_h, input_w = x.shape[2], x.shape[3]
        
        # 多尺度特征提取
        features = self.backbone(x)
        
        # 使用最后一个特征图
        x_feat = features[-1]
        
        # 特征融合并应用KAN增强
        enhanced_features = self.kan_module(x_feat)
        
        # 生成密度图
        density_map = self.density_head(enhanced_features)
        
        # 全局计数预测
        count = self.count_head(enhanced_features)
        
        # 多尺度融合
        final_density = self.multi_scale_processor(density_map, count)
        
        # 确保输出大小与期望一致
        if (input_h, input_w) != self.density_output_size:
            final_density = F.interpolate(
                final_density, 
                size=self.density_output_size, 
                mode='bilinear', 
                align_corners=False
            )
        
        return final_density, count

# 损失函数
class CompositionLoss(nn.Module):
    """组合损失函数，结合多种损失类型处理不同密度情况"""
    def __init__(self, lambda_density=1.0, lambda_count=1.0, lambda_ssim=0.1, lambda_persp=0.5):
        super(CompositionLoss, self).__init__()
        self.lambda_density = lambda_density  # 密度图MSE损失权重
        self.lambda_count = lambda_count      # 总数MAE损失权重
        self.lambda_ssim = lambda_ssim        # 结构相似性损失权重
        self.lambda_persp = lambda_persp      # 透视感知损失权重
        
        # SSIM损失 - 结构相似性
        self.ssim_loss = nn.MSELoss()  # 简化实现，实际中应使用真正的SSIM
        
        # 高斯加权MSE损失 - 对高密度区域给予更高权重
        self.weighted_mse = nn.MSELoss()  # 简化实现
        
        # 透视感知损失
        self.perspective_loss = nn.MSELoss()  # 简化实现
        
    def forward(self, pred_density, gt_density, pred_count, gt_count):
        # 密度图MSE损失
        density_loss = self.weighted_mse(pred_density, gt_density)
        
        # 计数绝对误差损失
        count_loss = F.l1_loss(pred_count, gt_count)
        
        # 结构相似性损失 - 简化实现
        ssim_loss = self.ssim_loss(pred_density, gt_density)
        
        # 透视感知损失 - 简化实现
        persp_loss = self.perspective_loss(pred_density, gt_density)
        
        # 组合损失
        total_loss = (self.lambda_density * density_loss + 
                      self.lambda_count * count_loss + 
                      self.lambda_ssim * ssim_loss +
                      self.lambda_persp * persp_loss)
        
        return total_loss, {
            'density_loss': density_loss.item(),
            'count_loss': count_loss.item(),
            'ssim_loss': ssim_loss.item(),
            'persp_loss': persp_loss.item(),
            'total_loss': total_loss.item()
        }

# 创建模型的工厂函数
def create_transformer_kan_crowd_counter(model_config=None):
    """
    创建Transformer-KAN人群计数模型
    
    参数:
        model_config: 模型配置字典，可选
        
    返回:
        创建的模型
    """
    # 默认配置
    default_config = {
        'backbone': 'swin_base_patch4_window7_224',
        'pretrained': False,  # 默认不使用预训练权重
        'use_kan': True,
        'density_output_size': (768, 768)
    }
    
    # 更新配置
    if model_config is not None:
        default_config.update(model_config)
    
    # 创建模型
    model = TransformerKANCrowdCounter(
        backbone=default_config['backbone'],
        pretrained=default_config['pretrained'],
        use_kan=default_config['use_kan'],
        density_output_size=default_config['density_output_size']
    )
    
    return model 