#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MobileKANCrowdCounter模型 - 用于轻量级高精度人群计数
结合了MobileNetV3和KAN（Kolmogorov-Arnold Network）的优势，
专为超高清人群密度估计设计，追求低计算成本和高准确率。
"""

import os
import sys
import math
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
import timm

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
            self.in_dim = in_dim if isinstance(in_dim, int) else in_dim[0]
            self.out_dim = in_dim if isinstance(in_dim, int) else in_dim[1]
            self.hid_dim = hid_dim
            self.depth = depth
            
            # 使用MLP替代KAN
            layers = [
                nn.Linear(self.in_dim, hid_dim),
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
            self.out_proj = nn.Linear(hid_dim, self.out_dim)
        
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
                x = x.view(*orig_shape)
            
            return x

# 轻量级KAN模块
class LightweightKANModule(nn.Module):
    """轻量级KAN模块 - 降低计算复杂度同时保持表达能力"""
    def __init__(self, in_channels, reduction=16, min_channels=128):
        super(LightweightKANModule, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.min_channels = min_channels
        
        # 输入投影 - 使用1x1卷积减少通道数
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, min_channels, kernel_size=1),
            nn.BatchNorm2d(min_channels),
            nn.ReLU(inplace=True)
        )
        
        # 简化的位置编码 - 固定尺寸以减少参数
        self.pos_embedding = nn.Parameter(torch.randn(1, min_channels, 32, 32))
        
        # KAN层或替代方案
        if HAS_KAN:
            self.kan_layers = nn.ModuleList([
                KAN([min_channels, min_channels]) 
                for _ in range(2)
            ])
        else:
            # 使用轻量级注意力替代
            self.dwconv_layers = nn.ModuleList([
                nn.Sequential(
                    # 深度可分离卷积
                    nn.Conv2d(min_channels, min_channels, kernel_size=3, padding=1, groups=min_channels),
                    nn.BatchNorm2d(min_channels),
                    nn.ReLU(inplace=True),
                    # 逐点卷积
                    nn.Conv2d(min_channels, min_channels, kernel_size=1),
                    nn.BatchNorm2d(min_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(2)
            ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(min_channels, min_channels, kernel_size=1),
            nn.BatchNorm2d(min_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        batch_size, _, h, w = x.shape
        
        # 输入投影
        x_proj = self.input_proj(x)
        
        # 添加位置编码
        pos_embed = F.interpolate(self.pos_embedding, size=(h, w), mode='bilinear', align_corners=False)
        x_pos = x_proj + pos_embed
        
        if HAS_KAN:
            # 使用KAN处理（分块操作以降低内存使用）
            x_kan = x_pos
            
            for kan_layer in self.kan_layers:
                # 我们使用分组处理来降低内存需求
                batch_size, channels, height, width = x_kan.shape
                block_size = 32  # 可以根据可用内存调整
                x_processed = torch.zeros_like(x_kan)
                
                for h_start in range(0, height, block_size):
                    h_end = min(h_start + block_size, height)
                    for w_start in range(0, width, block_size):
                        w_end = min(w_start + block_size, width)
                        
                        # 提取块
                        x_block = x_kan[:, :, h_start:h_end, w_start:w_end]
                        b, c, bh, bw = x_block.shape
                        
                        # 重塑为KAN可以处理的形式
                        x_flat = x_block.permute(0, 2, 3, 1).reshape(-1, c)
                        
                        # 应用KAN
                        out_flat = kan_layer(x_flat)
                        
                        # 重塑回原始形状
                        out_block = out_flat.reshape(b, bh, bw, c).permute(0, 3, 1, 2)
                        
                        # 更新结果
                        x_processed[:, :, h_start:h_end, w_start:w_end] = out_block
                
                # 残差连接
                x_kan = x_processed + x_pos
        else:
            # 使用深度可分离卷积处理
            x_kan = x_pos
            for layer in self.dwconv_layers:
                x_kan = layer(x_kan) + x_kan  # 残差连接
        
        # 输出投影
        out = self.output_proj(x_kan)
        
        return out

# KAN头部模块
class KANHead(nn.Module):
    """KAN头部模块，用于生成密度图"""
    def __init__(self, in_channels, hidden_channels=256, out_channels=1):
        super(KANHead, self).__init__()
        
        # KAN模块
        self.kan_module = LightweightKANModule(
            in_channels=in_channels,
            reduction=16,
            min_channels=hidden_channels
        )
        
        # 密度预测头
        self.density_head = LightDensityPredictionHead(
            in_channels=in_channels,
            hidden_channels=hidden_channels
        )
    
    def forward(self, x):
        # 应用KAN模块
        x = self.kan_module(x)
        
        # 生成密度图
        density_map = self.density_head(x)
        
        return density_map

# 轻量级密度预测头
class LightDensityPredictionHead(nn.Module):
    """轻量级密度图预测头"""
    
    def __init__(self, in_channels, hidden_channels=64, target_size=(384, 512)):
        super().__init__()
        
        # 保存目标尺寸
        self.target_size = target_size
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
        
        # 深度可分离卷积
        self.conv = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # 逐点卷积
            nn.Conv2d(in_channels, hidden_channels, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1)
        )
    
    def forward(self, x):
        # 应用通道注意力
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # 应用空间注意力
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights
        
        # 生成密度图
        density_map = self.conv(x)
        
        # 上采样到目标尺寸
        density_map = F.interpolate(
            density_map,
            size=self.target_size,
            mode='bilinear',
            align_corners=False
        )
        
        return density_map

# 轻量级全局计数头
class LightGlobalCountHead(nn.Module):
    """轻量级全局计数头，直接从特征预测总计数"""
    def __init__(self, in_channels, mid_channels):
        super(LightGlobalCountHead, self).__init__()
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 计数预测器 - 使用更少的隐藏单元
        self.count_predictor = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(mid_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 预测计数值（确保非负）
        count = self.count_predictor(x)
        count = F.relu(count)
        
        return count

# 多尺度处理器
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

# MobileKANCrowdCounter - 主模型类
class MobileKANCrowdCounter(nn.Module):
    """基于MobileNetV3和KAN的轻量级人群计数模型"""
    
    def __init__(self, backbone='mobilenetv3_large_100', pretrained=True, pretrained_path=None, dropout=0.0):
        super(MobileKANCrowdCounter, self).__init__()
        
        self.target_size = (384, 512)
        self.dropout_rate = dropout
        
        # 创建骨干网络
        if 'large' in backbone:
            self.backbone = mobilenet_v3_large(pretrained=False)
            self.feature_indices = [3, 6, 9, 12, 15]
            feature_dims = [24, 40, 80, 112, 160]
        else:
            self.backbone = mobilenet_v3_small(pretrained=False)
            self.feature_indices = [3, 6, 9, 12, 15]
            feature_dims = [16, 24, 40, 48, 96]
        
        # 特征转换层 - 统一通道数和初始化
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, 32, 1),  # 减少通道数
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ) for dim in feature_dims
        ])
        
        # 初始化权重
        self._initialize_weights()
        
        # 简化的注意力机制
        self.fusion_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 16, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            ) for _ in range(len(feature_dims))
        ])
        
        # 添加dropout层
        self.dropout = nn.Dropout2d(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity()
        
        # KAN模块
        self.kan = LightweightKANModule(
            in_channels=32 * len(feature_dims),
            reduction=4,  # 减小reduction
            min_channels=32
        )
        
        # 解码器 - 渐进式上采样
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout2d(self.dropout_rate) if self.dropout_rate > 0 else nn.Identity(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) for _ in range(3)
        ])
        
        # 输出层
        self.output_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(self.dropout_rate/2) if self.dropout_rate > 0 else nn.Identity(),
            nn.Conv2d(16, 1, 1),
            nn.ReLU(inplace=True),
            ScaleLayer(init_value=0.001)  # 进一步降低初始缩放因子
        )
        
        # 模型初始化完成后，尝试加载预训练权重
        if pretrained and pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path)
            
            # 检查是否是完整的训练状态
            if 'model_state_dict' in checkpoint:
                # 加载模型权重，而不是backbone权重
                print("检测到完整训练状态字典，加载model_state_dict...")
                if isinstance(self, nn.DataParallel):
                    # 使用strict=False允许部分加载权重，跳过不匹配的键
                    self.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
                else:
                    # 注意：只有在模型初始化完成后，这一步才能正常工作
                    # 使用strict=False允许部分加载权重，跳过不匹配的键
                    self.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("预训练模型部分加载成功! (仅加载匹配的层)")
            else:
                # 尝试直接加载backbone权重
                self.backbone.load_state_dict(checkpoint)
                print("仅加载backbone权重成功!")
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):
        # 特征提取
        features = []
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i in self.feature_indices:
                features.append(x)
        
        # 特征转换和归一化
        transformed_features = []
        base_size = features[-1].shape[2:]
        
        for feat, conv in zip(features, self.lateral_convs):
            # 特征归一化
            feat = F.layer_norm(feat, feat.shape[1:])
            # 转换通道数
            transformed = conv(feat)
            # 统一特征图大小
            if transformed.shape[2:] != base_size:
                transformed = F.interpolate(
                    transformed,
                    size=base_size,
                    mode='bilinear',
                    align_corners=False
                )
            transformed_features.append(transformed)
        
        # 应用注意力
        attended_features = []
        for feat, attention in zip(transformed_features, self.fusion_attention):
            att_weights = attention(feat)
            attended_features.append(feat * att_weights)
        
        # 特征融合
        fused_features = torch.cat(attended_features, dim=1)
        
        # 特征归一化
        fused_features = F.layer_norm(fused_features, fused_features.shape[1:])
        
        # 应用KAN
        refined_features = self.kan(fused_features)
        
        # 渐进式上采样
        x = refined_features
        for decoder_layer in self.decoder:
            x = decoder_layer(x)
            # 添加特征归一化
            x = F.layer_norm(x, x.shape[1:])
        
        # 生成最终输出
        density_map = self.output_conv(x)
        
        # 确保输出尺寸正确
        if density_map.shape[2:] != self.target_size:
            density_map = F.interpolate(
                density_map,
                size=self.target_size,
                mode='bilinear',
                align_corners=False
            )
        
        # 确保密度图非负且数值稳定
        density_map = F.relu(density_map)
        density_map = torch.clamp(density_map, min=0.0, max=100.0)  # 降低最大值限制
        
        return density_map

class ScaleLayer(nn.Module):
    """缩放层，用于调整输出范围"""
    def __init__(self, init_value=1.0):
        super(ScaleLayer, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
    
    def forward(self, x):
        return x * self.scale

# 损失函数
class CompositionLoss(nn.Module):
    """增强的组合损失函数，提高数值稳定性"""
    def __init__(self, lambda_density=1.0, lambda_count=1.0, lambda_ssim=0.1, lambda_persp=0.1):
        super(CompositionLoss, self).__init__()
        self.lambda_density = lambda_density
        self.lambda_count = lambda_count
        self.lambda_ssim = lambda_ssim  # 降低SSIM权重
        self.lambda_persp = lambda_persp  # 降低透视权重
        
        # MSE损失
        self.mse_loss = nn.MSELoss()
        
        # L1损失
        self.l1_loss = nn.L1Loss()
        
        # SSIM损失
        self.ssim_window_size = 11
        self.ssim_sigma = 1.5
        self.register_buffer('ssim_window', self._create_ssim_window())
        
    def _create_ssim_window(self):
        """创建SSIM计算用的高斯窗口"""
        window_1d = torch.exp(-torch.arange(-self.ssim_window_size//2 + 1, 
                                          self.ssim_window_size//2 + 1)**2 / (2*self.ssim_sigma**2))
        window_1d = window_1d / window_1d.sum()
        window_2d = window_1d.unsqueeze(0) * window_1d.unsqueeze(1)
        window = window_2d.unsqueeze(0).unsqueeze(0)
        return window
        
    def ssim_loss(self, pred, target):
        """计算SSIM损失"""
        c1, c2 = 0.01**2, 0.03**2
        
        # 使用卷积计算均值和方差
        mu_pred = F.conv2d(pred, self.ssim_window, padding=self.ssim_window_size//2)
        mu_target = F.conv2d(target, self.ssim_window, padding=self.ssim_window_size//2)
        
        mu_pred_sq = mu_pred.pow(2)
        mu_target_sq = mu_target.pow(2)
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.conv2d(pred * pred, self.ssim_window, padding=self.ssim_window_size//2) - mu_pred_sq
        sigma_target_sq = F.conv2d(target * target, self.ssim_window, padding=self.ssim_window_size//2) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, self.ssim_window, padding=self.ssim_window_size//2) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)) / \
                   ((mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2))
        
        return 1 - ssim_map.mean()
        
    def perspective_loss(self, pred, target, perspective_weights=None):
        """透视感知损失"""
        if perspective_weights is None:
            # 使用简单的距离权重
            h, w = pred.shape[2:]
            y_coords = torch.linspace(0, 1, h).view(-1, 1).repeat(1, w)
            perspective_weights = 1 + y_coords.to(pred.device)  # 远处(上方)给予更高权重
        
        weighted_loss = self.mse_loss(pred * perspective_weights, target * perspective_weights)
        return weighted_loss
        
    def forward(self, pred_density, gt_density, pred_count=None, gt_count=None):
        # 数值稳定性处理
        pred_density = torch.clamp(pred_density, min=0.0, max=100.0)
        gt_density = torch.clamp(gt_density, min=0.0, max=100.0)
        
        # 密度图损失 - 结合MSE和L1
        try:
            density_mse = self.mse_loss(pred_density, gt_density)
            density_l1 = self.l1_loss(pred_density, gt_density)
            density_loss = 0.5 * (density_mse + density_l1)
        except:
            print("Warning: 密度图损失计算出错，使用0")
            density_loss = torch.tensor(0.0).to(pred_density.device)
        
        # 计数损失
        try:
            if pred_count is not None and gt_count is not None:
                count_loss = self.l1_loss(pred_count, gt_count)
            else:
                pred_count = pred_density.sum(dim=(1,2,3))
                gt_count = gt_density.sum(dim=(1,2,3))
                count_loss = self.l1_loss(pred_count, gt_count)
        except:
            print("Warning: 计数损失计算出错，使用0")
            count_loss = torch.tensor(0.0).to(pred_density.device)
        
        # SSIM损失 - 添加数值稳定性
        try:
            ssim_loss = self.ssim_loss(pred_density, gt_density)
            ssim_loss = torch.clamp(ssim_loss, min=0.0, max=1.0)
        except:
            print("Warning: SSIM计算出错，使用MSE代替")
            ssim_loss = density_mse
        
        # 透视感知损失
        try:
            persp_loss = self.perspective_loss(pred_density, gt_density)
        except:
            print("Warning: 透视损失计算出错，使用L1代替")
            persp_loss = density_l1
        
        # 总损失 - 添加梯度裁剪
        total_loss = (self.lambda_density * density_loss + 
                     self.lambda_count * count_loss + 
                     self.lambda_ssim * ssim_loss +
                     self.lambda_persp * persp_loss)
        
        # 检查损失值是否合理
        if torch.isnan(total_loss) or torch.isinf(total_loss) or total_loss > 100:
            print(f"Warning: 异常损失值 - density_loss: {density_loss.item():.4f}, "
                  f"count_loss: {count_loss.item():.4f}, "
                  f"ssim_loss: {ssim_loss.item():.4f}, "
                  f"persp_loss: {persp_loss.item():.4f}")
            # 使用基础损失作为备选
            total_loss = density_loss
        
        return total_loss, {
            'density_loss': density_loss.item(),
            'count_loss': count_loss.item(),
            'ssim_loss': ssim_loss.item(),
            'persp_loss': persp_loss.item(),
            'total_loss': total_loss.item()
        }

# 创建模型的工厂函数
def create_mobile_kan_crowd_counter(config=None):
    """
    创建MobileKAN人群计数模型
    
    参数:
        config: 模型配置字典，包含以下可选参数：
            - backbone: 骨干网络类型，默认为'mobilenetv3_large_100'
            - pretrained: 是否使用预训练权重，默认为False
            - pretrained_path: 预训练权重路径，默认为None
    
    返回:
        model: MobileKANCrowdCounter模型实例
    """
    if config is None:
        config = {}
    
    # 默认配置
    default_config = {
        'backbone': 'mobilenetv3_large_100',
        'pretrained': False,
        'pretrained_path': None
    }
    
    # 更新配置
    default_config.update(config)
    config = default_config
    
    # 创建模型
    model = MobileKANCrowdCounter(
        backbone=config['backbone'],
        pretrained=config['pretrained'],
        pretrained_path=config['pretrained_path']
    )
    
    return model 