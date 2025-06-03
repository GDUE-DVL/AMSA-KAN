import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# 定义数据集类
class CrowdCountingDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        人群计数数据集
        Args:
            csv_file: 包含图像路径和计数的CSV文件
            transform: 图像变换
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        # 验证所有图像文件是否存在
        valid_indices = []
        for idx, row in self.data.iterrows():
            if os.path.exists(row['image_path']):
                valid_indices.append(idx)
            else:
                print(f"警告: 找不到图像文件 {row['image_path']}")
        
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        print(f"加载了 {len(self.data)} 个有效样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        count = self.data.iloc[idx]['count']
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(count, dtype=torch.float32)

# 定义网络模型
class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()
        # 使用预训练的VGG16作为基础
        vgg_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        
        self.frontend_feat = self._make_layers(vgg_layers, in_channels=3)
        
        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # 假设输入是 224x224 的图像
        x = self.frontend_feat(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x.squeeze(1)
    
    def _make_layers(self, cfg, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for images, counts in tqdm(train_loader, desc="训练"):
        images = images.to(device)
        counts = counts.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, counts)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# 验证函数
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    mae = 0.0
    
    with torch.no_grad():
        for images, counts in tqdm(val_loader, desc="验证"):
            images = images.to(device)
            counts = counts.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, counts)
            
            # 计算MAE
            mae += torch.abs(outputs - counts).sum().item()
            running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_mae = mae / len(val_loader.dataset)
    return epoch_loss, epoch_mae

# 测试函数
def test(model, test_loader, device):
    model.eval()
    mae = 0.0
    mse = 0.0
    
    all_preds = []
    all_counts = []
    
    with torch.no_grad():
        for images, counts in tqdm(test_loader, desc="测试"):
            images = images.to(device)
            counts = counts.to(device)
            
            # 前向传播
            outputs = model(images)
            
            # 记录预测结果
            all_preds.extend(outputs.cpu().numpy())
            all_counts.extend(counts.cpu().numpy())
            
            # 计算误差
            batch_mae = torch.abs(outputs - counts).sum().item()
            batch_mse = ((outputs - counts) ** 2).sum().item()
            
            mae += batch_mae
            mse += batch_mse
    
    # 计算平均MAE和MSE
    mae = mae / len(test_loader.dataset)
    mse = np.sqrt(mse / len(test_loader.dataset))
    
    # 绘制预测vs实际图
    plt.figure(figsize=(10, 7))
    plt.scatter(all_counts, all_preds, alpha=0.5)
    plt.plot([0, max(all_counts)], [0, max(all_counts)], 'r')
    plt.xlabel('实际人数')
    plt.ylabel('预测人数')
    plt.title(f'人群计数预测结果 (MAE={mae:.2f}, RMSE={mse:.2f})')
    plt.savefig('crowd_counting_test_results.png')
    
    return mae, mse

def main():
    parser = argparse.ArgumentParser(description='人群计数训练脚本')
    parser.add_argument('--train_csv', type=str, required=True, help='训练集CSV文件路径')
    parser.add_argument('--val_csv', type=str, required=True, help='验证集CSV文件路径')
    parser.add_argument('--test_csv', type=str, required=True, help='测试集CSV文件路径')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--output_dir', type=str, default='crowd_counting_results', help='输出目录')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用的GPU ID')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置GPU
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        # 限制内存使用
        torch.cuda.set_per_process_memory_fraction(0.7, args.gpu_id)  # 使用70%的GPU内存
        print(f"使用设备: {device} (CUDA {torch.cuda.get_device_name(args.gpu_id)})")
    else:
        device = torch.device('cpu')
        print(f"使用设备: {device}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    train_dataset = CrowdCountingDataset(args.train_csv, transform=transform)
    val_dataset = CrowdCountingDataset(args.val_csv, transform=transform)
    test_dataset = CrowdCountingDataset(args.test_csv, transform=transform)
    
    # 减少num_workers和批大小，以降低内存使用
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # 创建模型
    model = CrowdCounter().to(device)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 训练循环
    best_val_mae = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model.pth')
    train_losses = []
    val_losses = []
    val_maes = []
    
    for epoch in range(args.epochs):
        print(f"\n--- 第 {epoch+1}/{args.epochs} 轮 ---")
        
        # 训练
        train_loss = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # 验证
        val_loss, val_mae = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        
        # 更新学习率
        scheduler.step(val_mae)
        
        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证MAE: {val_mae:.4f}")
        
        # 保存最佳模型
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型, MAE = {val_mae:.4f}")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_maes, label='验证MAE')
    plt.xlabel('轮次')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    
    # 加载最佳模型并在测试集上评估
    model.load_state_dict(torch.load(best_model_path))
    test_mae, test_rmse = test(model, test_loader, device)
    
    print(f"\n测试结果:")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    
    # 保存测试结果
    with open(os.path.join(args.output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"MAE: {test_mae:.4f}\n")
        f.write(f"RMSE: {test_rmse:.4f}\n")

if __name__ == '__main__':
    main() 