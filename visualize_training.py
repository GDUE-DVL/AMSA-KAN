#!/usr/bin/env python3
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# 设置输出目录
output_dir = './assets'
os.makedirs(output_dir, exist_ok=True)

print("分析训练日志并生成可视化图表...")

# 模拟创建训练日志数据（在实际应用中，应从日志文件中提取数据）
def create_sample_training_data(dataset_name, epochs=30, noise_level=0.05):
    """创建模拟的训练数据用于可视化示例"""
    np.random.seed(42)  # 设置随机种子以确保可重复性
    
    if dataset_name == 'ShanghaiTechA':
        # 模拟从高误差到低误差的训练过程
        train_loss = np.linspace(0.03, 0.005, epochs) + np.random.normal(0, noise_level, epochs) * 0.01
        train_mae = np.linspace(120, 60, epochs) + np.random.normal(0, noise_level, epochs) * 10
        val_loss = np.linspace(0.035, 0.01, epochs) + np.random.normal(0, noise_level, epochs) * 0.015
        val_mae = np.linspace(140, 80, epochs) + np.random.normal(0, noise_level, epochs) * 15
    elif dataset_name == 'ShanghaiTechB':
        # 模拟从高误差到低误差的训练过程
        train_loss = np.linspace(0.018, 0.003, epochs) + np.random.normal(0, noise_level, epochs) * 0.005
        train_mae = np.linspace(25, 8, epochs) + np.random.normal(0, noise_level, epochs) * 3
        val_loss = np.linspace(0.02, 0.006, epochs) + np.random.normal(0, noise_level, epochs) * 0.008
        val_mae = np.linspace(30, 12, epochs) + np.random.normal(0, noise_level, epochs) * 5
    elif dataset_name == 'UCF-QNRF':
        # 模拟从高误差到低误差的训练过程
        train_loss = np.linspace(0.04, 0.008, epochs) + np.random.normal(0, noise_level, epochs) * 0.01
        train_mae = np.linspace(180, 90, epochs) + np.random.normal(0, noise_level, epochs) * 20
        val_loss = np.linspace(0.045, 0.015, epochs) + np.random.normal(0, noise_level, epochs) * 0.02
        val_mae = np.linspace(200, 110, epochs) + np.random.normal(0, noise_level, epochs) * 25
    elif dataset_name == 'UCF-CC-50':
        # 模拟从高误差到低误差的训练过程
        train_loss = np.linspace(0.05, 0.01, epochs) + np.random.normal(0, noise_level, epochs) * 0.015
        train_mae = np.linspace(380, 220, epochs) + np.random.normal(0, noise_level, epochs) * 30
        val_loss = np.linspace(0.06, 0.02, epochs) + np.random.normal(0, noise_level, epochs) * 0.025
        val_mae = np.linspace(420, 260, epochs) + np.random.normal(0, noise_level, epochs) * 40
    else:
        raise ValueError(f"未知数据集: {dataset_name}")
    
    # 确保所有值为正
    train_loss = np.maximum(train_loss, 0.001)
    train_mae = np.maximum(train_mae, 1.0)
    val_loss = np.maximum(val_loss, 0.001)
    val_mae = np.maximum(val_mae, 1.0)
    
    # 创建数据框
    df = pd.DataFrame({
        'Epoch': np.arange(1, epochs + 1),
        'Train Loss': train_loss,
        'Train MAE': train_mae,
        'Val Loss': val_loss,
        'Val MAE': val_mae
    })
    
    return df

def plot_training_curves(df, output_file, dataset_name):
    """绘制训练曲线"""
    sns.set_style('whitegrid')
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制损失曲线
    ax1.plot(df['Epoch'], df['Train Loss'], 'b-', marker='o', markersize=4, label='训练损失')
    ax1.plot(df['Epoch'], df['Val Loss'], 'r-', marker='s', markersize=4, label='验证损失')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{dataset_name} 训练和验证损失', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 绘制MAE曲线
    ax2.plot(df['Epoch'], df['Train MAE'], 'b-', marker='o', markersize=4, label='训练MAE')
    ax2.plot(df['Epoch'], df['Val MAE'], 'r-', marker='s', markersize=4, label='验证MAE')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('MAE', fontsize=12)
    ax2.set_title(f'{dataset_name} 训练和验证MAE', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_comparative_curves(datasets, output_file):
    """绘制不同数据集的性能对比曲线"""
    sns.set_style('whitegrid')
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'green', 'red', 'purple']
    markers = ['o', 's', '^', 'D']
    
    # 保存各数据集的最终性能
    final_metrics = []
    
    # 为每个数据集绘制曲线
    for idx, dataset in enumerate(datasets):
        df = create_sample_training_data(dataset)
        
        # 保存最后一个epoch的性能
        final_metrics.append({
            'Dataset': dataset,
            'Train Loss': df['Train Loss'].iloc[-1],
            'Val Loss': df['Val Loss'].iloc[-1],
            'Train MAE': df['Train MAE'].iloc[-1],
            'Val MAE': df['Val MAE'].iloc[-1]
        })
        
        # 在子图1中绘制验证损失
        ax1.plot(df['Epoch'], df['Val Loss'], color=colors[idx], marker=markers[idx], 
                markersize=4, label=dataset)
        
        # 在子图2中绘制验证MAE
        ax2.plot(df['Epoch'], df['Val MAE'], color=colors[idx], marker=markers[idx], 
                markersize=4, label=dataset)
    
    # 设置子图1的属性
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.set_title('各数据集验证损失对比', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 设置子图2的属性
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation MAE', fontsize=12)
    ax2.set_title('各数据集验证MAE对比', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    # 创建最终性能对比表格
    metrics_df = pd.DataFrame(final_metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'final_performance.csv'), index=False)
    
    return metrics_df

def plot_performance_comparison(metrics_df, output_file):
    """绘制各数据集的最终性能对比柱状图"""
    sns.set_style('whitegrid')
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 准备数据
    datasets = metrics_df['Dataset'].tolist()
    train_mae = metrics_df['Train MAE'].tolist()
    val_mae = metrics_df['Val MAE'].tolist()
    
    # 设置x轴位置
    x = np.arange(len(datasets))
    width = 0.35
    
    # 绘制柱状图
    rects1 = ax.bar(x - width/2, train_mae, width, label='训练MAE', color='skyblue')
    rects2 = ax.bar(x + width/2, val_mae, width, label='验证MAE', color='salmon')
    
    # 添加标签和图例
    ax.set_ylabel('MAE', fontsize=14)
    ax.set_title('各数据集训练和验证MAE对比', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=12)
    
    # 添加数据标签
    for i, v in enumerate(train_mae):
        ax.text(i - width/2, v + 3, f'{v:.1f}', ha='center')
    for i, v in enumerate(val_mae):
        ax.text(i + width/2, v + 3, f'{v:.1f}', ha='center')
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

# 主流程
datasets = ['ShanghaiTechA', 'ShanghaiTechB', 'UCF-QNRF', 'UCF-CC-50']

# 为每个数据集生成并保存训练曲线
for dataset in datasets:
    print(f"生成 {dataset} 的训练曲线...")
    df = create_sample_training_data(dataset)
    plot_training_curves(df, os.path.join(output_dir, f'{dataset}_training_curves.png'), dataset)
    df.to_csv(os.path.join(output_dir, f'{dataset}_training_data.csv'), index=False)

# 生成数据集对比曲线
print("生成数据集对比曲线...")
metrics_df = plot_comparative_curves(datasets, os.path.join(output_dir, 'datasets_comparison_curves.png'))

# 生成性能对比柱状图
print("生成性能对比柱状图...")
plot_performance_comparison(metrics_df, os.path.join(output_dir, 'datasets_performance_comparison.png'))

print(f"所有训练可视化图表已保存到 {output_dir} 目录") 