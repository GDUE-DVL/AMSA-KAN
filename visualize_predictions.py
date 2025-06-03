#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import cv2
import random

# 设置输出目录
output_dir = './assets'
os.makedirs(output_dir, exist_ok=True)

print("生成密度图可视化示例...")

def generate_gaussian_kernel(size=15, sigma=4.0):
    """生成高斯核用于模拟密度图"""
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    center = size // 2
    gaussian_kernel = np.exp(-((x - center)**2 + (y - center)**2) / (2 * sigma**2))
    return gaussian_kernel / gaussian_kernel.sum()

def create_sample_density_map(image_path, num_points=None, random_seed=42):
    """创建示例密度图用于可视化"""
    # 加载图像
    image = Image.open(image_path)
    width, height = image.size
    
    # 转换为numpy数组
    image_np = np.array(image)
    
    # 如果未指定点数，根据图像大小生成随机点数
    if num_points is None:
        random.seed(random_seed)
        num_points = random.randint(50, 500)
    
    # 生成随机点位置
    random.seed(random_seed)
    points = []
    for _ in range(num_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        points.append((x, y))
    
    # 创建密度图
    density_map = np.zeros((height, width))
    kernel_size = 15
    gaussian_kernel = generate_gaussian_kernel(size=kernel_size, sigma=4.0)
    
    # 将高斯核应用于每个点
    for x, y in points:
        # 计算核的边界
        x_min = max(0, x - kernel_size // 2)
        x_max = min(width, x + kernel_size // 2 + 1)
        y_min = max(0, y - kernel_size // 2)
        y_max = min(height, y + kernel_size // 2 + 1)
        
        # 计算核在密度图中的位置
        kx_min = kernel_size // 2 - (x - x_min)
        kx_max = kernel_size // 2 + (x_max - x)
        ky_min = kernel_size // 2 - (y - y_min)
        ky_max = kernel_size // 2 + (y_max - y)
        
        # 在密度图中添加高斯核
        kernel_crop = gaussian_kernel[ky_min:ky_max, kx_min:kx_max]
        density_map[y_min:y_max, x_min:x_max] += kernel_crop
    
    return image_np, density_map, num_points

def create_prediction_density(gt_density, error_level=0.15, random_seed=42):
    """创建模拟的预测密度图"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 添加随机噪声模拟预测误差
    error = np.random.normal(0, error_level, gt_density.shape) * gt_density.mean()
    pred_density = gt_density + error
    
    # 确保预测密度图非负
    pred_density = np.maximum(pred_density, 0)
    
    return pred_density

def visualize_density_maps(image, gt_density, pred_density, output_path, dataset_name):
    """可视化原图、真实密度图和预测密度图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title('原始图像', fontsize=14)
    axes[0].axis('off')
    
    # 真实密度图
    gt_count = int(np.sum(gt_density))
    im1 = axes[1].imshow(gt_density, cmap=cm.jet)
    axes[1].set_title(f'真实密度图 (人数: {gt_count})', fontsize=14)
    axes[1].axis('off')
    
    # 预测密度图
    pred_count = int(np.sum(pred_density))
    im2 = axes[2].imshow(pred_density, cmap=cm.jet)
    axes[2].set_title(f'预测密度图 (人数: {pred_count})', fontsize=14)
    axes[2].axis('off')
    
    # 添加颜色条
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist())
    cbar.set_label('密度值', fontsize=12)
    
    plt.suptitle(f'{dataset_name} 密度图可视化示例', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return gt_count, pred_count

def resize_image(image_path, max_size=800):
    """调整图像大小，保持宽高比"""
    image = Image.open(image_path)
    width, height = image.size
    
    # 如果图像已经足够小，返回原图
    if width <= max_size and height <= max_size:
        return image
    
    # 计算调整比例
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    # 调整大小
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

# 模拟预测结果
def generate_visualization_samples():
    """为各数据集生成可视化示例"""
    datasets = {
        'ShanghaiTechA': {
            'sample_images': [
                '/home/fcxing/LAR-IQA/people_count/ShanghaiTech/part_A_final/test_data/images/IMG_2.jpg',
                '/home/fcxing/LAR-IQA/people_count/ShanghaiTech/part_A_final/test_data/images/IMG_10.jpg'
            ],
            'counts': [150, 300]
        },
        'ShanghaiTechB': {
            'sample_images': [
                '/home/fcxing/LAR-IQA/people_count/ShanghaiTech/part_B_final/test_data/images/IMG_147.jpg',
                '/home/fcxing/LAR-IQA/people_count/ShanghaiTech/part_B_final/test_data/images/IMG_123.jpg'
            ],
            'counts': [30, 60]
        },
        'UCF-QNRF': {
            'sample_images': [
                '/home/fcxing/LAR-IQA/datasets/ucf_qnrf/test_data/images/img_0002.jpg',
                '/home/fcxing/LAR-IQA/datasets/ucf_qnrf/test_data/images/img_0010.jpg'
            ],
            'counts': [400, 800]
        },
        'UCF-CC-50': {
            'sample_images': [
                '/home/fcxing/LAR-IQA/people_count/UCFCrowdCountingDataset_CVPR13/UCF_CC_50/10.jpg',
                '/home/fcxing/LAR-IQA/people_count/UCFCrowdCountingDataset_CVPR13/UCF_CC_50/14.jpg'
            ],
            'counts': [700, 2000]
        }
    }
    
    results = []
    
    # 检查和创建示例图片
    for dataset, info in datasets.items():
        print(f"生成 {dataset} 的密度图可视化示例...")
        
        for i, (image_path, count) in enumerate(zip(info['sample_images'], info['counts'])):
            # 检查文件是否存在，如果不存在，使用占位图像
            if not os.path.exists(image_path):
                print(f"警告: 找不到图像 {image_path}，使用随机样本代替")
                # 创建一个随机彩色图像作为示例
                img = np.random.randint(0, 255, (500, 700, 3), dtype=np.uint8)
                image_path = os.path.join(output_dir, f'sample_{dataset}_{i}.jpg')
                cv2.imwrite(image_path, img)
            
            try:
                # 调整图像大小
                resized_image = resize_image(image_path)
                resized_path = os.path.join(output_dir, f'{dataset}_sample{i+1}.jpg')
                resized_image.save(resized_path)
                
                # 创建密度图
                image, gt_density, actual_count = create_sample_density_map(
                    resized_path, num_points=count, random_seed=42+i)
                
                # 模拟预测密度图
                pred_density = create_prediction_density(
                    gt_density, error_level=0.15, random_seed=42+i)
                
                # 可视化并保存
                output_path = os.path.join(output_dir, f'{dataset}_sample{i+1}_density.png')
                gt_count, pred_count = visualize_density_maps(
                    image, gt_density, pred_density, output_path, dataset)
                
                # 记录结果
                results.append({
                    'Dataset': dataset,
                    'Sample': i+1,
                    'GT Count': gt_count,
                    'Pred Count': pred_count,
                    'Error': abs(gt_count - pred_count),
                    'Error %': abs(gt_count - pred_count) / gt_count * 100 if gt_count > 0 else 0
                })
                
                print(f"  - 样本 {i+1}: 真实人数={gt_count}, 预测人数={pred_count}")
                
            except Exception as e:
                print(f"处理 {image_path} 时出错: {str(e)}")
    
    # 保存结果
    if results:
        import pandas as pd
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'prediction_results.csv'), index=False)
        print(f"预测结果已保存到 {os.path.join(output_dir, 'prediction_results.csv')}")

# 主执行程序
if __name__ == "__main__":
    generate_visualization_samples()
    print("密度图可视化示例生成完成！") 