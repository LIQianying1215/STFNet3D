import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from SeismicNet import SeismicNet
import torch
import time
import argparse
import json
from pathlib import Path

# 早停法类
class EarlyStopping:
    def __init__(self, patience=10, verbose=True, delta=0, save_path='best_model.pth'):
        """
        Args:
            patience: 验证集损失不再改善的等待轮数
            verbose: 是否打印早停信息
            delta: 最小改善阈值
            save_path: 最佳模型保存路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        
    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model, epoch):
        """保存最佳模型"""
        if self.verbose:
            print(f'验证损失减小 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 保存模型...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss
        }, self.save_path)
        self.val_loss_min = val_loss

# 模拟退火学习率调度器
class SimulatedAnnealingLR:
    def __init__(self, optimizer, T_max=100, T_min=1e-4, alpha=0.95):
        """
        模拟退火学习率调度器
        Args:
            optimizer: 优化器
            T_max: 初始温度
            T_min: 最小温度
            alpha: 退火率
        """
        self.optimizer = optimizer
        self.T_max = T_max
        self.T_min = T_min
        self.alpha = alpha
        self.T = T_max
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        """更新学习率"""
        if self.T > self.T_min:
            self.T *= self.alpha
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * (self.T / self.T_max)
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']

# 学习率预热调度器
class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, initial_lr, target_lr):
        """
        学习率预热调度器
        Args:
            optimizer: 优化器
            warmup_steps: 预热步数
            initial_lr: 初始学习率
            target_lr: 目标学习率
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.current_step = 0
        
    def step(self):
        """更新学习率"""
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']

def normalize_data(data, min_val, max_val):
    """
    对给定的数据进行最小-最大归一化。
    """
    if max_val - min_val == 0:
        return data
    return (data - min_val) / (max_val - min_val)

def find_labels_ranges(label_folder, label_count=3):
    """找到标签数据的范围"""
    min_vals = [float('inf')] * label_count
    max_vals = [float('-inf')] * label_count
    
    for entry in os.listdir(label_folder):
        if entry.startswith('model_parameters') and entry.endswith('.txt'):
            label_file_path = os.path.join(label_folder, entry)
            with open(label_file_path, 'r') as file:
                file.readline()  # 跳过第一行（参数名称）
                labels_data = file.readline().strip().split()
                labels_data = [float(x) for x in labels_data]
                
                for j in range(label_count):
                    min_vals[j] = min(min_vals[j], labels_data[j])
                    max_vals[j] = max(max_vals[j], labels_data[j])
    return min_vals, max_vals

def find_data_range(data_folder):
    """找到输入数据的范围"""
    min_val = float('inf')
    max_val = float('-inf')
    
    for entry in os.listdir(data_folder):
        if entry.startswith('MR') and entry.endswith('.txt'):
            input_file_path = os.path.join(data_folder, entry)
            with open(input_file_path, 'r') as file:
                data = file.readline().strip().split()
                data = [float(x) for x in data]
                min_val = min(min_val, min(data))
                max_val = max(max_val, max(data))
    return min_val, max_val

def load_data_from_split(split_folder, target_length=3001, label_count=3, subset='train'):
    """
    从划分的文件夹中加载指定子集的数据

    Args:
        split_folder: 划分文件夹路径（如 '702010/'）
        target_length: 目标数据长度
        label_count: 标签数量
        subset: 要加载的子集，可选 'train', 'val', 'test'
    """
    # 根据子集参数动态确定MR和parameters的文件夹名
    if subset == 'train':
        mr_subdir = 'MR'
        param_subdir = 'parameters'
    elif subset == 'val':
        mr_subdir = 'val_MR'
        param_subdir = 'val_parameters'
    elif subset == 'test':
        mr_subdir = 'test_MR'
        param_subdir = 'test_parameters'
    else:
        raise ValueError("subset 参数必须是 'train', 'val' 或 'test'")

    # 使用动态确定的文件夹名拼接路径
    mr_folder = os.path.join(split_folder, mr_subdir)
    param_folder = os.path.join(split_folder, param_subdir)

    # ... 函数其余部分（查找文件、读取数据、归一化等）保持不变 ...
    print(f"正在从以下路径加载 '{subset}' 子集数据:")
    print(f"  - MR 文件目录: {mr_folder}")
    print(f"  - Parameters 文件目录: {param_folder}")

    # 确保目录存在
    if not os.path.exists(mr_folder):
        raise FileNotFoundError(f"MR 目录不存在: {mr_folder}")
    if not os.path.exists(param_folder):
        raise FileNotFoundError(f"Parameters 目录不存在: {param_folder}")
    
    input_data = []
    labels = []
    file_indices = []  # 保存文件索引
    # 找到数据范围
    min_val, max_val = find_data_range(mr_folder)
    labels_min_vals, labels_max_vals = find_labels_ranges(param_folder, label_count)
    
    # 加载MR数据
    for entry in sorted(os.listdir(mr_folder)):
        if entry.startswith('MR') and entry.endswith('.txt'):
            # 提取文件索引
            idx = entry[2:-4]  # 去掉'MR'和'.txt'
            file_indices.append(idx)

            input_file_path = os.path.join(mr_folder, entry)
            with open(input_file_path, 'r') as file:
                data = file.readline().strip().split()
                data = [float(x) for x in data]
                
                # 数据预处理
                data = [min_val] + data
                if len(data) < target_length:
                    data += [min_val] * (target_length - len(data))
                
                data = normalize_data(np.array(data), min_val, max_val)
                input_data.append(data)
    
    # 加载parameters数据
    param_data = []
    for entry in sorted(os.listdir(param_folder)):
        if entry.startswith('model_parameters') and entry.endswith('.txt'):
            
            # 检查对应的MR文件是否存在
            mr_file = entry.replace('model_parameters', 'MR')
            if not os.path.exists(os.path.join(mr_folder, mr_file)):
                continue
                
            label_file_path = os.path.join(param_folder, entry)
            with open(label_file_path, 'r') as file:
                file.readline()  # 跳过第一行
                labels_data = file.readline().strip().split()
                labels_data = [float(x) for x in labels_data]
                
                # 分别归一化每个标签
                normalized_labels = [normalize_data(np.array([labels_data[j]]), 
                                                  labels_min_vals[j], labels_max_vals[j])[0] 
                                   for j in range(label_count)]
                param_data.append(normalized_labels)

    if len(input_data) != len(param_data):
        print(f"警告: 输入数据数量({len(input_data)})与标签数据数量({len(param_data)})不匹配")
        min_len = min(len(input_data), len(param_data))
        input_data = input_data[:min_len]
        param_data = param_data[:min_len]
        file_indices = file_indices[:min_len]

    input_data = np.array(input_data)
    labels = np.array(param_data).reshape(-1, label_count)
    
    return (torch.tensor(input_data, dtype=torch.float32), 
            torch.tensor(labels, dtype=torch.float32), 
            labels_min_vals, labels_max_vals, file_indices)

def set_seed(seed):
    """固定随机数种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def cosine_annealing_schedule(t, lr_max, lr_min, T_cycle):
    """余弦退火学习率调度"""
    t = t % T_cycle
    lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T_cycle))
    return lr_t

def update_learning_rate(optimizer, lr):
    """更新优化器的学习率"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def denormalize_label_predictions(predictions, min_vals, max_vals):
    """
    对预测结果进行反归一化
    """
    denormalized_predictions = []
    for pred in predictions:
        pred_denorm = pred.clone()
        for i in range(len(pred)):
            pred_denorm[i] = (pred[i] * (max_vals[i] - min_vals[i])) + min_vals[i]
        denormalized_predictions.append(pred_denorm)
    
    return torch.stack(denormalized_predictions)

def calculate_tu_te(mus_values, dc_values):
    """
    计算Tu和Te参数
    """
    Tu = mus_values / (120 * 10**12)
    Te = dc_values / Tu
    return Tu, Te

def save_individual_prediction_files(predictions, file_indices, results_dir, labels_min_vals, labels_max_vals):
    """
    将每个预测结果保存为单独的txt文件
    """
    pred_dir = os.path.join(results_dir, 'individual_predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    # 反归一化预测结果
    predictions_denorm = denormalize_label_predictions(predictions, labels_min_vals, labels_max_vals)
    
    for i, (pred, idx) in enumerate(zip(predictions_denorm, file_indices)):
        filename = f"pred_model_parameters{idx}.txt"
        filepath = os.path.join(pred_dir, filename)
        
        with open(filepath, 'w') as f:
            # 写入参数名称（与原始文件格式一致）
            f.write("mu_s Dc tau\n")
            # 写入预测的参数值
            f.write(f"{pred[0]:.6f} {pred[1]:.6f} {pred[2]:.6f}\n")
    
    print(f"已保存 {len(predictions)} 个预测文件到 {pred_dir}")
    return pred_dir
def create_comprehensive_visualizations(predictions_tensor, labels_tensor, results_dir, labels_min_vals, labels_max_vals):
    """
    创建全面的可视化图表
    """
    
    # 设置全局字体大小
    plt.rcParams.update({'font.size': 14})  # 增大基础字体大小[1,3]
   
    # 反归一化预测结果
    predictions_denorm = denormalize_label_predictions(predictions_tensor, labels_min_vals, labels_max_vals)
    labels_denorm = denormalize_label_predictions(labels_tensor, labels_min_vals, labels_max_vals)
    
    predictions = predictions_denorm.numpy()
    true_labels = labels_denorm.numpy()
    
    # 创建基于组别的颜色映射
    num_groups = labels_tensor.shape[0]
    group_indices = np.arange(num_groups)
    norm = plt.Normalize(0, num_groups - 1)
    cmap = plt.cm.viridis
    
    # 计算真实值和预测值的Tu和Te
    Tu_true, Te_true = calculate_tu_te(true_labels[:, 0], true_labels[:, 1])  # μs是第一个参数，Dc是第二个
    Tu_pre, Te_pre = calculate_tu_te(predictions[:, 0], predictions[:, 1])
    
    # 1. 创建Tu和Te的对角线图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Tu参数的对角线图
    axes[0, 0].scatter(Tu_true, Tu_pre, c=group_indices, cmap=cmap, norm=norm, alpha=0.6, s=8)
    min_tu = min(Tu_true.min(), Tu_pre.min())
    max_tu = max(Tu_true.max(), Tu_pre.max())
    axes[0, 0].plot([min_tu, max_tu], [min_tu, max_tu], 'k--', alpha=0.5, linewidth=2)
    axes[0, 0].grid(True, linestyle='--', alpha=0.3)
    r2_tu = np.corrcoef(Tu_true, Tu_pre)[0, 1] ** 2
    axes[0, 0].text(0.05, 0.95, f'R² = {r2_tu:.4f}', transform=axes[0, 0].transAxes,
                    bbox=dict(facecolor='white', alpha=0.9), fontsize=16)
    axes[0, 0].set_title('Predicted vs True T$_u$', fontsize=18)
    axes[0, 0].set_xlabel('True T$_u$', fontsize=16)
    axes[0, 0].set_ylabel('Predicted T$_u$', fontsize=16)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=14) 
    
    # Te参数的对角线图
    axes[0, 1].scatter(Te_true, Te_pre, c=group_indices, cmap=cmap, norm=norm, alpha=0.6, s=8)
    min_te = min(Te_true.min(), Te_pre.min())
    max_te = max(Te_true.max(), Te_pre.max())
    axes[0, 1].plot([min_te, max_te], [min_te, max_te], 'k--', alpha=0.5, linewidth=2)
    axes[0, 1].grid(True, linestyle='--', alpha=0.3)
    r2_te = np.corrcoef(Te_true, Te_pre)[0, 1] ** 2
    axes[0, 1].text(0.05, 0.95, f'R² = {r2_te:.4f}', transform=axes[0, 1].transAxes,
                    bbox=dict(facecolor='white', alpha=0.9), fontsize=16)
    axes[0, 1].set_title('Predicted vs True T$_e$', fontsize=18)  # 使用下标
    axes[0, 1].set_xlabel('True T$_e$', fontsize=16)
    axes[0, 1].set_ylabel('Predicted T$_e$', fontsize=16)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=14)
    
    # Tu误差分布图
    tu_errors = (Tu_pre - Tu_true) / Tu_true * 100  # 百分比误差
    axes[1, 0].hist(tu_errors, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    axes[1, 0].set_title('T$_u$ Prediction Error Distribution', fontsize=18)
    axes[1, 0].set_xlabel('Relative Error (%)', fontsize=16)
    axes[1, 0].set_ylabel('Frequency', fontsize=16)
    axes[1, 0].grid(True, linestyle='--', alpha=0.3)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=14)
    
    # Te误差分布图
    te_errors = (Te_pre - Te_true) / Te_true * 100  # 百分比误差
    axes[1, 1].hist(te_errors, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    axes[1, 1].set_title('T$_e$ Prediction Error Distribution', fontsize=18)
    axes[1, 1].set_xlabel('Relative Error (%)', fontsize=16)
    axes[1, 1].set_ylabel('Frequency', fontsize=16)
    axes[1, 1].grid(True, linestyle='--', alpha=0.3)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tu_te_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 创建三个参数的对比图 - 使用正确的符号和单位
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 调整图形尺寸
    
    # 使用正确的符号：μs（保持），Dc→D_c，tau→τ
    label_symbols = [r'$\mu_s$', r'$D_c$ (m)', r'$\tau$ (MPa)']  # 添加单位[8,9](@ref)
    label_names = ['μs', 'Dc', 'tau']  # 保留原始名称用于标题
    
    for i in range(3):
        true_values = true_labels[:, i]
        pred_values = predictions[:, i]
        
        scatter = axes[i].scatter(true_values, pred_values, 
                                c=group_indices, 
                                cmap=cmap,
                                norm=norm, 
                                alpha=0.6,
                                s=8)  # 增大点的大小
        
        min_val = min(true_values.min(), pred_values.min())
        max_val = max(true_values.max(), pred_values.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
        axes[i].grid(True, linestyle='--', alpha=0.3)
        
        r2 = np.corrcoef(true_values, pred_values)[0, 1] ** 2
        axes[i].text(0.05, 0.95, f'R² = {r2:.4f}', 
                    transform=axes[i].transAxes,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'),
                    fontsize=16,  # 增大文本字体
                    verticalalignment='top')
        
        axes[i].set_title(f'Predicted vs True {label_symbols[i]}', fontsize=18)
        axes[i].set_xlabel(f'True {label_symbols[i]}', fontsize=16)
        axes[i].set_ylabel(f'Predicted {label_symbols[i]}', fontsize=16)
        axes[i].tick_params(axis='both', which='major', labelsize=14)
        
        if i == 2:
            cbar = plt.colorbar(scatter, ax=axes[i])
            cbar.set_label('Sample Index', fontsize=14)
            cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'parameter_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 创建参数差值分布图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i in range(3):
        true_values = true_labels[:, i]
        pred_values = predictions[:, i]
        diff_values = pred_values - true_values
        
        max_abs_diff = max(abs(diff_values.min()), abs(diff_values.max()))
        bin_edges = np.linspace(-max_abs_diff, max_abs_diff, 31)
        
        axes[i].hist(diff_values, bins=bin_edges, color='skyblue', edgecolor='black', alpha=0.7)
        axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
        axes[i].set_title(f'Predicted - True {label_symbols[i]}', fontsize=18)
        axes[i].set_xlabel(f'Difference ({label_symbols[i].split(" ")[-1]})', fontsize=16)  # 添加单位
        axes[i].set_ylabel('Frequency', fontsize=16)
        axes[i].grid(True, linestyle='--', alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'parameter_difference_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 创建实际参数值的分布图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i in range(3):
        true_values = true_labels[:, i]
        axes[i].hist(true_values, bins=15, color='skyblue', edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Distribution of {label_symbols[i]}', fontsize=18)
        axes[i].set_xlabel(label_symbols[i], fontsize=16)  # 使用带单位的标签
        axes[i].set_ylabel('Frequency', fontsize=16)
        axes[i].grid(True, linestyle='--', alpha=0.3)
        axes[i].tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'parameter_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存统计信息
    save_statistical_analysis(predictions, true_labels, Tu_true, Tu_pre, Te_true, Te_pre, results_dir, label_names)
    
    # 恢复默认设置，避免影响其他图形
    plt.rcParams.update({'font.size': 10})
    
    return predictions, true_labels

def save_statistical_analysis(predictions, true_labels, Tu_true, Tu_pre, Te_true, Te_pre, results_dir, label_n):
    """
    保存详细的统计分析结果
    """
    # Tu和Te的统计信息
    with open(os.path.join(results_dir, 'tu_te_statistics.txt'), 'w') as f:
        f.write("=== Tu and Te Parameter Statistics ===\n")
        f.write(f"Tu_true - Min: {Tu_true.min():.6e}, Max: {Tu_true.max():.6e}, Mean: {Tu_true.mean():.6e}\n")
        f.write(f"Tu_pre - Min: {Tu_pre.min():.6e}, Max: {Tu_pre.max():.6e}, Mean: {Tu_pre.mean():.6e}\n")
        f.write(f"Te_true - Min: {Te_true.min():.6e}, Max: {Te_true.max():.6e}, Mean: {Te_true.mean():.6e}\n")
        f.write(f"Te_pre - Min: {Te_pre.min():.6e}, Max: {Te_pre.max():.6e}, Mean: {Te_pre.mean():.6e}\n")
        
        # 计算误差统计
        tu_mae = np.mean(np.abs(Tu_pre - Tu_true))
        tu_rmse = np.sqrt(np.mean((Tu_pre - Tu_true) ** 2))
        r2_tu = np.corrcoef(Tu_true, Tu_pre)[0, 1] ** 2
        
        te_mae = np.mean(np.abs(Te_pre - Te_true))
        te_rmse = np.sqrt(np.mean((Te_pre - Te_true) ** 2))
        r2_te = np.corrcoef(Te_true, Te_pre)[0, 1] ** 2
        
        f.write(f"\nTu - MAE: {tu_mae:.6e}, RMSE: {tu_rmse:.6e}, R²: {r2_tu:.6f}\n")
        f.write(f"Te - MAE: {te_mae:.6e}, RMSE: {te_rmse:.6e}, R²: {r2_te:.6f}\n")
    
    # 相对误差统计
    with open(os.path.join(results_dir, 'relative_error_stats.txt'), 'w') as f:
        f.write("=== Relative Error Statistics ===\n")
        for i in range(3):
            true_values = true_labels[:, i]
            pred_values = predictions[:, i]
            relative_errors = np.abs(pred_values - true_values) / np.abs(true_values) * 100
            
            mean_error = np.mean(relative_errors)
            max_error = np.max(relative_errors)
            min_error = np.min(relative_errors)
            
            f.write(f"\n{label_n[i]} Parameter:\n")
            f.write(f"Mean Relative Error: {mean_error:.2f}%\n")
            f.write(f"Max Relative Error: {max_error:.2f}%\n")
            f.write(f"Min Relative Error: {min_error:.2f}%\n")
    
    # 保存Tu和Te数据
    tu_te_data = np.column_stack((Tu_true, Tu_pre, Te_true, Te_pre))
    np.savetxt(os.path.join(results_dir, 'tu_te_data.txt'), tu_te_data, 
               header='Tu_true Tu_pre Te_true Te_pre', fmt='%.6e')
    
    print("所有统计信息已保存完成")

def test_model(model, test_loader, device, results_dir, labels_min_vals, labels_max_vals, test_file_indices):
    """
    测试模型并保存结果
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_inputs, batch_labels in test_loader:
            batch_inputs = batch_inputs.unsqueeze(1).to(device)
            predictions = model(batch_inputs)
            
            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())
    
    # 合并所有批次的预测结果
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 保存单个预测文件
    pred_dir = save_individual_prediction_files(all_predictions, test_file_indices, results_dir, labels_min_vals, labels_max_vals)
    
    # 创建全面的可视化
    predictions, true_labels = create_comprehensive_visualizations(all_predictions, all_labels, results_dir, labels_min_vals, labels_max_vals)
    
    # 计算Dc参数的相对误差并保存典型案例
    save_selected_cases(predictions, true_labels, results_dir)
    
    # 保存测试结果总结
    results_summary = {
        'prediction_files_dir': pred_dir,
        'total_test_samples': len(all_predictions),
        'labels_min_vals': labels_min_vals,
        'labels_max_vals': labels_max_vals
    }
    
    with open(os.path.join(results_dir, 'test_results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print(f"\n=== 测试完成 ===")
    print(f"测试样本数量: {len(all_predictions)}")
    print(f"预测文件保存到: {pred_dir}")
    
    return all_predictions, all_labels

def save_selected_cases(predictions, true_labels, results_dir):
    """
    保存选定案例（高、中、低误差）
    """
    # 计算Dc参数的相对误差
    dc_true = true_labels[:, 1]
    dc_pred = predictions[:, 1]
    dc_relative_errors = np.abs((dc_pred - dc_true) / dc_true)
    
    # 创建索引数组并根据相对误差排序
    indices = np.arange(len(dc_relative_errors))
    sorted_indices = indices[np.argsort(dc_relative_errors)]
    
    # 选择案例
    high_error_indices = sorted_indices[-10:]
    mid_point = len(sorted_indices) // 2
    medium_error_indices = sorted_indices[mid_point-5 : mid_point+5]
    low_error_indices = sorted_indices[:10]
    
    # 保存选定的数据
    with open(os.path.join(results_dir, 'selected_cases.txt'), 'w') as f:
        f.write("=== High Error Cases ===\n")
        for idx in high_error_indices:
            true_vals = true_labels[idx]
            pred_vals = predictions[idx]
            rel_error = dc_relative_errors[idx] * 100
            f.write(f"True: {true_vals[0]:.6f} {true_vals[1]:.6f} {true_vals[2]:.6f} | ")
            f.write(f"Pred: {pred_vals[0]:.6f} {pred_vals[1]:.6f} {pred_vals[2]:.6f} | ")
            f.write(f"Error: {rel_error:.2f}%\n")
        
        f.write("\n=== Medium Error Cases ===\n")
        for idx in medium_error_indices:
            true_vals = true_labels[idx]
            pred_vals = predictions[idx]
            rel_error = dc_relative_errors[idx] * 100
            f.write(f"True: {true_vals[0]:.6f} {true_vals[1]:.6f} {true_vals[2]:.6f} | ")
            f.write(f"Pred: {pred_vals[0]:.6f} {pred_vals[1]:.6f} {pred_vals[2]:.6f} | ")
            f.write(f"Error: {rel_error:.2f}%\n")
        
        f.write("\n=== Low Error Cases ===\n")
        for idx in low_error_indices:
            true_vals = true_labels[idx]
            pred_vals = predictions[idx]
            rel_error = dc_relative_errors[idx] * 100
            f.write(f"True: {true_vals[0]:.6f} {true_vals[1]:.6f} {true_vals[2]:.6f} | ")
            f.write(f"Pred: {pred_vals[0]:.6f} {pred_vals[1]:.6f} {pred_vals[2]:.6f} | ")
            f.write(f"Error: {rel_error:.2f}%\n")

def train_model(config):
    """主训练函数"""
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 创建结果保存目录
    results_dir = f"{config['results_base_dir']}/{config['split_name']}_{config['experiment_name']}"
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存配置
    with open(f"{results_dir}/config.json", 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"开始训练: {config['split_name']} - {config['experiment_name']}")
    print(f"结果保存到: {results_dir}")
    
    # 加载数据
    split_folder = f"{config['data_base_dir']}/{config['split_name']}"
   # 加载训练数据 (使用默认的 subset='train')
    train_inputs, train_labels, labels_min_val, labels_max_val, _ = load_data_from_split(split_folder, config['target_length'], config['label_count'] # subset 默认为 'train'
)

    # 加载验证数据 (明确指定 subset='val')
    val_inputs, val_labels, _, _, _= load_data_from_split(split_folder, config['target_length'], config['label_count'], subset='val')
   
    # 创建DataLoader
    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(val_inputs, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型定义
    model = SeismicNet().to(device)
    criterion = nn.MSELoss()
    
    # 优化器选择
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], 
                             weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], 
                              weight_decay=config['weight_decay'])
    else:  # SGD
        optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'], 
                            momentum=0.9, weight_decay=config['weight_decay'])
    
    # 学习率调度器
    if config['lr_scheduler'] == 'cosine':
        lr_each_epoch = [cosine_annealing_schedule(t, config['learning_rate'], 
                                                  config['lr_min'], config['T_cycle']) 
                        for t in range(config['num_epochs'])]
    elif config['lr_scheduler'] == 'simulated_annealing':
        scheduler = SimulatedAnnealingLR(optimizer, T_max=100, T_min=1e-4, alpha=0.95)
    elif config['lr_scheduler'] == 'warmup':
        scheduler = WarmupScheduler(optimizer, warmup_steps=config['warmup_steps'],
                                  initial_lr=config['lr_min'], target_lr=config['learning_rate'])
    
    # 早停法
    early_stopping = EarlyStopping(
        patience=config['patience'], 
        verbose=True,
        save_path=f"{results_dir}/best_model.pth"
    )
    
    # 混合精度训练[1,2](@ref)
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] else None
    
    # 训练记录
    train_losses = []
    val_losses = []
    learning_rates = []
    
    best_val_loss = float('inf')
    
    print("开始训练循环...")
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        # 更新学习率
        if config['lr_scheduler'] == 'cosine':
            lr = lr_each_epoch[epoch]
            update_learning_rate(optimizer, lr)
        elif config['lr_scheduler'] in ['simulated_annealing', 'warmup']:
            scheduler.step()
            lr = scheduler.get_lr()
        else:  # 固定学习率
            lr = config['learning_rate']
        
        learning_rates.append(lr)
        
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_inputs, batch_labels in train_loader:
            batch_inputs = batch_inputs.unsqueeze(1).to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            if config['use_amp'] and scaler is not None:
                # 混合精度训练[1](@ref)
                with torch.cuda.amp.autocast():
                    predictions = model(batch_inputs)
                    loss = criterion(predictions, batch_labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_labels in val_loader:
                batch_inputs = batch_inputs.unsqueeze(1).to(device)
                batch_labels = batch_labels.to(device)
                
                predictions = model(batch_inputs)
                loss = criterion(predictions, batch_labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 早停检查
        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print("早停触发!")
            break
        
        # 更新最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, f"{results_dir}/best_val_model.pth")
        
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{config["num_epochs"]}, '
              f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, '
              f'LR: {lr:.6f}, Time: {epoch_time:.2f}s')
    
    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"{results_dir}/final_model.pth")
    
    # 绘制损失曲线和学习率曲线
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(learning_rates)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_curves.png")
    plt.close()
    
    # 保存损失数据
    with open(f"{results_dir}/losses.txt", 'w') as f:
        for i, (train_loss, val_loss, lr) in enumerate(zip(train_losses, val_losses, learning_rates)):
            f.write(f"{i}, {train_loss}, {val_loss}, {lr}\n")
    
    print(f"训练完成! 最佳验证损失: {best_val_loss:.6f}")

        # 测试模型
    print("\n开始测试模型...")
    test_inputs, test_labels, test_labels_min_vals, test_labels_max_vals, test_file_indices = load_data_from_split(
        split_folder, config['target_length'], config['label_count'], subset='test'
    )
    
    test_dataset = TensorDataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # 加载最佳模型进行测试
    best_model = SeismicNet().to(device)
    checkpoint = torch.load(f"{results_dir}/best_val_model.pth")
    best_model.load_state_dict(checkpoint['model_state_dict'])
    
    test_predictions, test_labels = test_model(
        best_model, test_loader, device, results_dir, 
        test_labels_min_vals, test_labels_max_vals, test_file_indices
    )
    
    print(f"\n训练和测试完成! 所有结果保存在: {results_dir}")

    return best_val_loss

def main():
    """主函数：运行不同配置的实验"""
    
    # 基础配置
    base_config = {
        'data_base_dir': '/Nas/Liqy/EarthquakeSeismicNet/stratified_splits',
        'results_base_dir': '/Nas/Liqy/EarthquakeSeismicNet/results',
        'target_length': 3001,
        'label_count': 3,
        'num_epochs': 500,
        'patience': 50,
        'seed': 42,
        'weight_decay': 1e-5,
        'use_amp': True,  # 使用混合精度训练[1](@ref)
    }
    
    # 定义实验配置
    experiments = []
    
    # 不同数据划分
    # splits = ['751510']
    splits = ['702010', '751510', '801010']
    
    # 不同学习率
    # learning_rates = [1e-3]
    learning_rates = [1e-2, 1e-3, 1e-4]
    
    # 不同批次大小
    # batch_sizes = [64]
    batch_sizes = [64, 128, 256]

    # 不同学习率调度器
    # lr_schedulers = ['cosine']
    lr_schedulers = ['fixed', 'cosine', 'simulated_annealing']
    
    # 生成实验配置
    for split in splits:
        for lr in learning_rates:
            for batch_size in batch_sizes:
                for lr_scheduler in lr_schedulers:
                    experiment_name = f"split_{split}_lr_{lr}_bs_{batch_size}_sched_{lr_scheduler}"
                    
                    config = base_config.copy()
                    config.update({
                        'split_name': split,
                        'experiment_name': experiment_name,
                        'learning_rate': lr,
                        'lr_min': lr / 10,
                        'batch_size': batch_size,
                        'lr_scheduler': lr_scheduler,
                        'T_cycle': 100,
                        'warmup_steps': 50,
                        'optimizer': 'AdamW',  # 使用AdamW优化器[3](@ref)
                    })
                    
                    experiments.append(config)
    
    print(f"总共 {len(experiments)} 个实验配置")
    
    # 运行实验
    results = []
    for i, config in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"运行实验 {i+1}/{len(experiments)}")
        print(f"配置: {config['experiment_name']}")
        print(f"{'='*60}")
        
        try:
            best_val_loss = train_model(config)
            results.append({
                'config': config,
                'best_val_loss': best_val_loss,
                'status': 'success'
            })
        except Exception as e:
            print(f"实验失败: {e}")
            results.append({
                'config': config,
                'best_val_loss': float('inf'),
                'status': 'failed',
                'error': str(e)
            })
    
    # 保存实验结果汇总
    results_summary = []
    for result in results:
        if result['status'] == 'success':
            results_summary.append({
                'experiment': result['config']['experiment_name'],
                'best_val_loss': result['best_val_loss'],
                'split': result['config']['split_name'],
                'learning_rate': result['config']['learning_rate'],
                'batch_size': result['config']['batch_size'],
                'lr_scheduler': result['config']['lr_scheduler']
            })
    
    # 按验证损失排序
    results_summary.sort(key=lambda x: x['best_val_loss'])
    
    # 保存结果汇总
    with open(f"{base_config['results_base_dir']}/experiments_summary.csv", 'w') as f:
        f.write("Experiment,Split,LearningRate,BatchSize,Scheduler,BestValLoss\n")
        for result in results_summary:
            f.write(f"{result['experiment']},{result['split']},{result['learning_rate']},"
                   f"{result['batch_size']},{result['lr_scheduler']},{result['best_val_loss']:.6f}\n")
    
    # 打印最佳结果
    print("\n最佳实验结果:")
    for i, result in enumerate(results_summary[:5]):
        print(f"{i+1}. {result['experiment']}: {result['best_val_loss']:.6f}")

if __name__ == "__main__":
    main()