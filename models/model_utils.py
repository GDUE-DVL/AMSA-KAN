"""
模型工具类，提供常用的损失函数和评估指标实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CombinedLoss(nn.Module):
    """组合损失函数，结合MSE和MAE以及可选的其他损失函数"""
    def __init__(self, mse_weight=0.5, mae_weight=0.5, smooth_l1_weight=0.0, density_weight=0.0):
        super(CombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.smooth_l1_weight = smooth_l1_weight
        self.density_weight = density_weight
        
        # 确保权重总和为1
        total_weight = mse_weight + mae_weight + smooth_l1_weight + density_weight
        if abs(total_weight - 1.0) > 1e-6:
            self.mse_weight /= total_weight
            self.mae_weight /= total_weight
            self.smooth_l1_weight /= total_weight
            self.density_weight /= total_weight
        
        # 定义损失函数
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
    
    def forward(self, pred, target):
        """计算组合损失值"""
        loss = 0
        
        # MSE损失
        if self.mse_weight > 0:
            mse = self.mse_loss(pred, target)
            loss += self.mse_weight * mse
        
        # MAE损失
        if self.mae_weight > 0:
            mae = self.mae_loss(pred, target)
            loss += self.mae_weight * mae
        
        # Smooth L1损失
        if self.smooth_l1_weight > 0:
            smooth_l1 = self.smooth_l1_loss(pred, target)
            loss += self.smooth_l1_weight * smooth_l1
        
        # 密度图损失 - 针对密度图的特殊损失
        if self.density_weight > 0:
            # 计算密度图上的均方误差（考虑局部区域的一致性）
            density_loss = self.compute_density_loss(pred, target)
            loss += self.density_weight * density_loss
        
        return loss
    
    def compute_density_loss(self, pred, target):
        """计算密度图相关的损失
        使用局部感知损失，关注局部区域的一致性
        """
        # 计算局部区域的平均密度
        # 使用平均池化对局部区域进行整合
        kernel_size = 4
        local_avg_pred = F.avg_pool2d(pred, kernel_size=kernel_size, stride=kernel_size)
        local_avg_target = F.avg_pool2d(target, kernel_size=kernel_size, stride=kernel_size)
        
        # 计算局部区域的MSE
        local_mse = self.mse_loss(local_avg_pred, local_avg_target)
        
        # 计算全局总和差异
        pred_sum = torch.sum(pred, dim=(1, 2, 3), keepdim=True)
        target_sum = torch.sum(target, dim=(1, 2, 3), keepdim=True)
        global_diff = torch.abs(pred_sum - target_sum).mean()
        
        # 结合局部MSE和全局差异
        return local_mse + 0.1 * global_diff

class FocalLoss(nn.Module):
    """Focal Loss用于解决样本不平衡问题"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        """计算Focal Loss"""
        # 计算平方误差
        se = (pred - target) ** 2
        
        # 添加Focal Loss的权重
        weight = torch.abs(pred - target) ** self.gamma
        
        # 计算最终损失
        focal_loss = self.alpha * weight * se
        
        return torch.mean(focal_loss)

class SSIMLoss(nn.Module):
    """结构相似性损失(SSIM Loss)"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)
    
    def forward(self, pred, target):
        """计算SSIM损失"""
        (_, channel, _, _) = pred.size()
        
        if channel == self.channel and self.window.data.type() == pred.data.type():
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            
            if pred.is_cuda:
                window = window.cuda(pred.get_device())
            window = window.type_as(pred)
            
            self.window = window
            self.channel = channel
        
        return 1 - self._ssim(pred, target, window, self.window_size, channel, self.size_average)
    
    def _ssim(self, pred, target, window, window_size, channel, size_average=True):
        """计算SSIM值"""
        mu1 = F.conv2d(pred, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def _create_window(self, window_size, channel):
        """创建SSIM计算用的高斯窗口"""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _gaussian(self, window_size, sigma):
        """生成一维高斯分布"""
        gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss / gauss.sum()

def compute_metrics(pred, target):
    """计算评估指标"""
    # 将张量转换为numpy数组
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # 确保是一维数组
    pred = pred.flatten()
    target = target.flatten()
    
    # 计算MAE
    mae = np.mean(np.abs(pred - target))
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    
    # 计算相对误差(Relative Error)
    epsilon = 1e-6  # 避免除以零
    rel_err = np.mean(np.abs(pred - target) / (target + epsilon))
    
    # 计算准确率(Accuracy) - 相对误差小于0.1的比例
    accuracy = np.mean((np.abs(pred - target) / (target + epsilon)) < 0.1)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'rel_err': rel_err,
        'accuracy': accuracy
    }

class AdvancedCombinedLoss(nn.Module):
    """高级组合损失函数，结合MSE、MAE、SSIM和Focal Loss"""
    def __init__(self, mse_weight=0.55, mae_weight=0.40, ssim_weight=0.05, 
                 focal_gamma=2.0, local_coherence_weight=0.1):
        super(AdvancedCombinedLoss, self).__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.ssim_weight = ssim_weight
        self.focal_gamma = focal_gamma
        self.local_coherence_weight = local_coherence_weight
        
        # 归一化权重
        total_weight = mse_weight + mae_weight + ssim_weight
        if abs(total_weight - 1.0) > 1e-6:
            self.mse_weight /= total_weight
            self.mae_weight /= total_weight
            self.ssim_weight /= total_weight
        
        # 定义损失函数组件
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.mae_loss = nn.L1Loss(reduction='mean')
        self.ssim_loss = SSIMLoss(window_size=11)
        
    def forward(self, pred, target):
        """计算高级组合损失"""
        loss = 0
        
        # 基础损失组件
        if self.mse_weight > 0:
            mse = self.mse_loss(pred, target)
            loss += self.mse_weight * mse
            
        if self.mae_weight > 0:
            mae = self.mae_loss(pred, target)
            loss += self.mae_weight * mae
            
        if self.ssim_weight > 0:
            ssim = self.ssim_loss(pred, target)
            loss += self.ssim_weight * ssim
            
        # 添加Focal Loss组件 - 更关注难样本
        if self.focal_gamma > 0:
            # 计算平方误差
            se = (pred - target) ** 2
            # 添加权重以关注高误差区域
            weight = torch.abs(pred - target) ** self.focal_gamma
            focal_loss = (weight * se).mean()
            loss += 0.1 * focal_loss  # 较小的权重
            
        # 添加局部一致性损失
        if self.local_coherence_weight > 0:
            local_loss = self.compute_local_coherence_loss(pred, target)
            loss += self.local_coherence_weight * local_loss
            
        return loss
    
    def compute_local_coherence_loss(self, pred, target):
        """计算局部一致性损失，确保预测的密度图在局部区域上与真实密度图保持一致"""
        # 使用卷积计算局部梯度
        kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0).to(pred.device)
        kernel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0).to(pred.device)
        
        # 扩展卷积核到所有通道
        b, c, h, w = pred.shape
        kernel_x = kernel_x.repeat(c, 1, 1, 1)
        kernel_y = kernel_y.repeat(c, 1, 1, 1)
        
        # 计算梯度
        pred_grad_x = F.conv2d(pred, kernel_x, padding=1, groups=c)
        pred_grad_y = F.conv2d(pred, kernel_y, padding=1, groups=c)
        target_grad_x = F.conv2d(target, kernel_x, padding=1, groups=c)
        target_grad_y = F.conv2d(target, kernel_y, padding=1, groups=c)
        
        # 计算梯度差异
        loss_grad_x = F.mse_loss(pred_grad_x, target_grad_x)
        loss_grad_y = F.mse_loss(pred_grad_y, target_grad_y)
        
        return loss_grad_x + loss_grad_y

# 修改损失函数工厂方法
def get_loss_function(loss_type, **kwargs):
    """根据损失函数类型创建损失函数实例"""
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mae':
        return nn.L1Loss()
    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    elif loss_type == 'focal':
        return FocalLoss(alpha=kwargs.get('alpha', 1), gamma=kwargs.get('gamma', 2))
    elif loss_type == 'ssim':
        return SSIMLoss(window_size=kwargs.get('window_size', 11))
    elif loss_type == 'combined':
        return CombinedLoss(
            mse_weight=kwargs.get('mse_weight', 0.5),
            mae_weight=kwargs.get('mae_weight', 0.5),
            smooth_l1_weight=kwargs.get('smooth_l1_weight', 0.0),
            density_weight=kwargs.get('density_weight', 0.0)
        )
    elif loss_type == 'advanced_combined':
        return AdvancedCombinedLoss(
            mse_weight=kwargs.get('mse_weight', 0.55),
            mae_weight=kwargs.get('mae_weight', 0.40),
            ssim_weight=kwargs.get('ssim_weight', 0.05),
            focal_gamma=kwargs.get('focal_gamma', 2.0),
            local_coherence_weight=kwargs.get('local_coherence_weight', 0.1)
        )
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

# 学习率调度器工厂方法
def get_lr_scheduler(optimizer, scheduler_type, **kwargs):
    """根据调度器类型创建学习率调度器实例"""
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 10),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [30, 60, 90]),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('t_max', 100),
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_type == 'warmup_cosine':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=kwargs.get('total_steps', 100),
            pct_start=kwargs.get('warmup_ratio', 0.1),
            anneal_strategy='cos',
            div_factor=kwargs.get('div_factor', 25),
            final_div_factor=kwargs.get('final_div_factor', 10000)
        )
    else:
        raise ValueError(f"不支持的学习率调度器类型: {scheduler_type}") 