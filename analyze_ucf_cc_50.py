#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置输出目录
output_dir = './assets'
os.makedirs(output_dir, exist_ok=True)

print("Analyzing UCF-CC-50 dataset and generating visualizations...")

# 读取CSV文件
print("Reading data files...")
train_path = './datasets/ucf_cc_50/train.csv'
val_path = './datasets/ucf_cc_50/val.csv'
test_path = './datasets/ucf_cc_50/test.csv'

try:
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    # 添加数据集标识
    train_df['split'] = 'Train'
    val_df['split'] = 'Validation'
    test_df['split'] = 'Test'
    
    # 合并数据集
    all_data = pd.concat([train_df, val_df, test_df])
    
    print(f"Total images: {len(all_data)}")
    print(f"Train: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")
    print(f"Test: {len(test_df)} images")
    
    # 1. Histogram
    print("Generating count distribution histogram...")
    plt.figure(figsize=(12, 8))
    sns.set_style('whitegrid')
    sns.histplot(data=all_data, x='count', hue='split', bins=20, kde=True, 
                 palette={'Train': 'blue', 'Validation': 'orange', 'Test': 'green'})
    plt.title('UCF-CC-50 Count Distribution', fontsize=16)
    plt.xlabel('People Count', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'ucf_cc_50_distribution.png'), dpi=300)
    plt.close()
    
    # 2. Boxplot
    print("Generating boxplot...")
    plt.figure(figsize=(10, 8))
    sns.boxplot(data=all_data, x='split', y='count', palette={'Train': 'blue', 'Validation': 'orange', 'Test': 'green'})
    plt.title('UCF-CC-50 Count by Split', fontsize=16)
    plt.xlabel('Split', fontsize=14)
    plt.ylabel('People Count', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'ucf_cc_50_boxplot.png'), dpi=300)
    plt.close()
    
    # 3. Statistics
    print("Calculating statistics...")
    splits = ['Train', 'Validation', 'Test', 'All']
    stats = []
    
    for split in splits[:-1]:
        subset = all_data[all_data['split'] == split]
        stats.append({
            'Split': split,
            'Count': len(subset),
            'Min': subset['count'].min(),
            'Max': subset['count'].max(),
            'Mean': subset['count'].mean(),
            'Std': subset['count'].std()
        })
    
    # 添加总体统计信息
    stats.append({
        'Split': 'All',
        'Count': len(all_data),
        'Min': all_data['count'].min(),
        'Max': all_data['count'].max(),
        'Mean': all_data['count'].mean(),
        'Std': all_data['count'].std()
    })
    
    # 创建统计表格
    stats_df = pd.DataFrame(stats)
    print(stats_df.round(2))
    
    # 保存统计信息
    stats_df.to_csv(os.path.join(output_dir, 'ucf_cc_50_statistics.csv'), index=False)
    
    # 创建统计表格图片
    print("Generating statistics table image...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    table = plt.table(
        cellText=stats_df.values.round(2),
        colLabels=stats_df.columns,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ucf_cc_50_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Pie chart
    print("Generating split pie chart...")
    plt.figure(figsize=(10, 10))
    split_counts = all_data['split'].value_counts()
    plt.pie(split_counts, labels=split_counts.index, autopct='%1.1f%%', 
            colors=['blue', 'orange', 'green'], startangle=140)
    plt.axis('equal')
    plt.title('UCF-CC-50 Split Proportion', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'ucf_cc_50_split_pie.png'), dpi=300)
    plt.close()
    
    # 5. Model comparison
    print("Generating model comparison bar chart...")
    # 以下为模拟数据，在实际应用中应替换为真实的模型性能数据
    models = ['LAR-IQA (ours)', 'Model A', 'Model B', 'Model C']
    mae_values = [219.5, 250.3, 241.2, 260.8]
    mse_values = [316.9, 340.2, 334.7, 380.1]
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, mae_values, width, label='MAE', color='steelblue')
    rects2 = ax.bar(x + width/2, mse_values, width, label='MSE', color='firebrick')
    
    # 添加标签和图例
    ax.set_ylabel('Error Value', fontsize=14)
    ax.set_title('UCF-CC-50 Model Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=12)
    
    # 添加数据标签
    for i, v in enumerate(mae_values):
        ax.text(i - width/2, v + 5, f'{v:.1f}', ha='center')
    for i, v in enumerate(mse_values):
        ax.text(i + width/2, v + 5, f'{v:.1f}', ha='center')
    
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ucf_cc_50_models_comparison.png'), dpi=300)
    plt.close()
    
    print(f"All figures saved to {output_dir}")
    
except Exception as e:
    print(f"Error occurred: {str(e)}") 