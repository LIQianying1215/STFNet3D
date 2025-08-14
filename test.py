import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from SeismicNet import SeismicNet

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def normalize_data(data, min_val, max_val):
    """
    对给定的数据进行最小-最大归一化。
    """
    # 避免除以零的情况，当最小值和最大值相等时
    if max_val - min_val == 0:
        return data
    return (data - min_val) / (max_val - min_val)

def find_labels_ranges(label, label_count):
    min_vals = [float('inf')] * label_count
    max_vals = [float('-inf')] * label_count
    entries = os.listdir(label)
    for entry in entries:
        if os.path.isfile(os.path.join(label, entry)):
            label_file_path = os.path.join(label, entry)
            with open(label_file_path, 'r') as file:
                file.readline()  # 跳过第一行（参数名称）
                labels_data = file.readline().strip().split()
                labels_data = [float(x) for x in labels_data]

                for j in range(label_count):
                    min_vals[j] = min(min_vals[j], labels_data[j])
                    max_vals[j] = max(max_vals[j], labels_data[j])
    return min_vals, max_vals


def find_data_range(data_path):
    """
    扫描数据集以找到最小值和最大值。
    """
    min_val = float('inf')
    max_val = float('-inf')
    entries = os.listdir(data_path)
    for entry in entries:
        if os.path.isfile(os.path.join(data_path, entry)):
            input_file_path = os.path.join(data_path, entry)
            with open(input_file_path, 'r') as file:
                data = file.readline().strip().split()
                data = [float(x) for x in data]
                min_val = min(min_val, min(data))
                max_val = max(max_val, max(data))
    return min_val, max_val


def denormalize_label_predictions(predictions, min_vals, max_vals):
    """
    对批量预测结果的每个标签分别进行反归一化。

    参数:
    - predictions: 预测结果，为一个包含多个批次预测的列表，每个批次预测为一个32x3的tensor。
    - min_vals: 每个标签的最小值列表。
    - max_vals: 每个标签的最大值列表。

    返回:
    - denormalized_predictions: 反归一化后的预测结果列表。
    """
    denormalized_predictions = []
    for pred in predictions:
        # print(pred)
        # pred = np.array(pred)
        # pred现在是32x3的tensor，直接对每个标签进行反归一化
        for i in range(3):  # 对3个标签分别进行处理
            # print(pred.shape)
            pred[i] = (pred[i] * (max_vals[i] - min_vals[i])) + min_vals[i]
        denormalized_predictions.append(pred)
        denormalized_predictions_tensor = torch.stack(denormalized_predictions)
    # 由于predictions已经是一个列表，直接返回处理后的列表即可

    return denormalized_predictions_tensor

# 模型目录和存放测试结果的目录
traindirectory = '/Nas/Liqy/EarthquakeSeismicNet/MR/result250728/'
directory = '/Nas/Liqy/EarthquakeSeismicNet/MR/TestResult250728/'
# 加载模型
model = SeismicNet()
filename = '250728_1_500.pth'
model_path = f'{traindirectory}{filename}'

# 加载保存的字典
checkpoint = torch.load(model_path)

# 从保存的字典中提取模型状态字典并加载到模型中
model.load_state_dict(checkpoint['model_state_dict'])

# model.load_state_dict(torch.load(model_path))
model.eval()  # 设置为评估模式

# 数据目录 改为自己的数据目录
input_folder = r'/Nas/Liqy/EarthquakeSeismicNet/data_only_90/test_MR'
label_folder = r'/Nas/Liqy/EarthquakeSeismicNet/data_only_90/test_parameter'

# 使用find_data_range和find_labels_ranges函数找到数据和标签的范围

min_val, max_val = find_data_range(input_folder)
labels_min_vals, labels_max_vals = find_labels_ranges(label_folder, 3)

# 生成所有数据点的索引
total_files = len(os.listdir(input_folder))  # 假设每个文件夹中的文件数相同
input_files = sorted(os.listdir(input_folder))
label_files = sorted(os.listdir(label_folder))

inputs = []
labels = []
fault_positions = []  # 用于存储断层位置信息

# 创建输入文件索引到文件名的映射
input_map = {}

for input_file in input_files:
    if input_file.startswith('MR') and input_file.endswith('.txt'):
        # 从 'MR1234.txt' 提取索引 '1234'
        idx = input_file[2:-4]  # 去掉 'MR' 和 '.txt'
        input_map[idx] = input_file

# 处理每个标签文件并找到对应的输入文件
for label_file in label_files:
    if label_file.startswith('model_parameters') and label_file.endswith('.txt'):
        idx = label_file[16:-4]
        
        if idx in input_map:
            input_file = input_map[idx]
            
            # 处理输入数据
            input_file_path = os.path.join(input_folder, input_file)
            with open(input_file_path, 'r') as file:
                data = np.array([float(x) for x in file.readline().strip().split()])
                # 在前方补一个最小值，确保其归一化后能变为从0开始
                data = np.insert(data, 0, min_val)

                # # 找到第一个小于最大值/400的值
                # max_data = max(data)
                # threshold = max_val / 400

                # # 找到第一个满足条件的索引
                # zero_idx = None
                # for i , val in enumerate(data[1:]):
                #     if abs(val) < threshold:
                #         zero_idx = i
                #         break               
                # # 如果找到这样的位置，将该位置后的所有值设为0
                # if zero_idx is not None:
                #     data[zero_idx:] = [0.0] * len(data[zero_idx:])

                current_length = data.shape[0]
                padding_length = 3001 - current_length
                if padding_length > 0:
                    data = np.pad(data, (0, padding_length), mode='constant', constant_values=min_val)
                normalized_data = normalize_data(data, min_val, max_val)
                inputs.append(normalized_data)
            
            # 处理标签数据，现在读取5个参数
            label_file_path = os.path.join(label_folder, label_file)
            with open(label_file_path, 'r') as file:
                file.readline()  # 跳过第一行
                labels_data = file.readline().strip().split()
                # 前三个参数作为标签
                labels.append([float(labels_data[j]) for j in range(3)])
                # 后两个参数作为断层位置
                fault_positions.append([float(labels_data[j]) for j in range(3, 5)])
            
            # 打印进度（每100个文件）
            if len(inputs) % 100 == 0:
                print(f"Processed {len(inputs)} pairs of files...")
        else:
            print(f"Warning: No matching input file for label file {label_file}")

# 最终验证
print(f"Total processed file pairs: {len(inputs)}")
if len(inputs) == 0:
    raise ValueError("No matching file pairs found!")
if len(inputs) != len(labels):
    raise ValueError(f"Mismatch between number of inputs ({len(inputs)}) and labels ({len(labels)})")

# print(labels)
# 转换为PyTorch张量并添加必要的维度
inputs = np.array(inputs)
labels = np.array(labels)
inputs_tensor = torch.tensor(inputs, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
labels_tensor = torch.tensor(labels, dtype=torch.float32)

torch.save(labels_tensor, 'la_tensor.pt')
# 使用模型进行预测
with torch.no_grad():
    predictions_tensor = model(inputs_tensor)
predictions_tensor1 = denormalize_label_predictions(predictions_tensor, labels_min_vals, labels_max_vals)

torch.save(predictions_tensor, 'pre_tensor.pt')

num_groups = labels_tensor.shape[0]
group_indices = np.arange(num_groups)
#创建基于组别的颜色映射
norm = plt.Normalize(0, num_groups - 1)
cmap = plt.cm.viridis

# 创建三个子图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
label_n = ['μs', 'Dc', 'tau']

# 绘制每个参数的对比图
for i in range(3):

    # 获取真实值和预测值
    true_values = labels_tensor[:, i].numpy()
    pred_values = predictions_tensor[:, i].numpy()
    
    # # 计算散点图的颜色值（基于真实值）
    # norm = plt.Normalize(true_values.min(), true_values.max())
    # colors = plt.cm.viridis(norm(true_values))
    
    # 绘制散点图
    scatter = axes[i].scatter(true_values, pred_values, 
                            c=group_indices, 
                            cmap=cmap,
                            norm=norm, 
                            alpha=0.6,
                            s=5)  # 这里设置点的大小，可以调整这个值
    
    # 添加对角线
    min_val = min(true_values.min(), pred_values.min())
    max_val = max(true_values.max(), pred_values.max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # 添加网格线
    axes[i].grid(True, linestyle='--', alpha=0.3)
    
    # 计算R²值
    r2 = np.corrcoef(true_values, pred_values)[0, 1] ** 2
    
    # 在图中添加R²值（左上角）
    axes[i].text(0.05, 0.95, f'R² = {r2:.4f}', 
                transform=axes[i].transAxes,  # 使用轴的相对坐标
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),  # 添加白色背景框
                fontsize=10,
                verticalalignment='top')
    
    # 设置标题和标签
    axes[i].set_title(f'Predicted vs True {label_n[i]}')
    axes[i].set_xlabel(f'True {label_n[i]}')
    axes[i].set_ylabel(f'Pred {label_n[i]}')
    
    # 添加颜色条(仅在第3子图显示，避免重复)
    if i == 2:
        cbar = plt.colorbar(scatter, ax=axes[i])
        cbar.set_label('Sample Index')
 
    # 计算并打印统计信息
    mse = np.mean((true_values - pred_values) ** 2)
    print(f'{label_n[i]} - MSE: {mse:.6f}, R²: {r2:.6f}')

# 调整子图之间的间距
plt.tight_layout()

# 保存图像
plt.savefig(os.path.join(directory,'prediction_comparison24115.png'), dpi=300, bbox_inches='tight')
plt.show()

# 创建三个子图之前，先计算并输出相对误差统计信息
print("\n=== Relative Error Statistics ===")

for i in range(3):
    true_values = labels_tensor[:, i].numpy()
    pred_values = predictions_tensor[:, i].numpy()
    
    # 计算相对误差 (以百分比形式显示)
    relative_errors = np.abs(pred_values - true_values) / np.abs(true_values) * 100
    
    # 计算统计量
    mean_error = np.mean(relative_errors)
    max_error = np.max(relative_errors)
    min_error = np.min(relative_errors)
    
    print(f"\n{label_n[i]} Parameter:")
    print(f"Mean Relative Error: {mean_error:.2f}%")
    print(f"Max Relative Error: {max_error:.2f}%")
    print(f"Min Relative Error: {min_error:.2f}%")

# 将统计结果保存到文件
with open(os.path.join(directory,'relative_error_stats.txt'), 'w') as f:
    f.write("=== Relative Error Statistics ===\n")
    for i in range(3):
        true_values = labels_tensor[:, i].numpy()
        pred_values = predictions_tensor[:, i].numpy()
        relative_errors = np.abs(pred_values - true_values) / np.abs(true_values) * 100
        
        mean_error = np.mean(relative_errors)
        max_error = np.max(relative_errors)
        min_error = np.min(relative_errors)
        
        f.write(f"\n{label_n[i]} Parameter:\n")
        f.write(f"Mean Relative Error: {mean_error:.2f}%\n")
        f.write(f"Max Relative Error: {max_error:.2f}%\n")
        f.write(f"Min Relative Error: {min_error:.2f}%\n")

# 创建参数差值分布图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
label_n = ['μs', 'Dc', 'tau']

for i in range(3):
    true_values = labels_tensor[:, i].numpy()
    pred_values = predictions_tensor[:, i].numpy()
    
    # 计算差值
    diff_values = pred_values - true_values
    
    # 计算对称的直方图范围
    max_abs_diff = max(abs(diff_values.min()), abs(diff_values.max()))
    bin_edges = np.linspace(-max_abs_diff, max_abs_diff, 31)  # 30个bins，31个边界
    
    # 绘制直方图
    axes[i].hist(diff_values, bins=bin_edges, color='skyblue', edgecolor='black')
    
    # 添加竖直的零线
    axes[i].axvline(x=0, color='red', linestyle='--', alpha=0.5)
    
    # 设置标题和标签
    axes[i].set_title(f'pred_{label_n[i]} - true_{label_n[i]}')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    
    # 添加网格线
    axes[i].grid(True, linestyle='--', alpha=0.3)

# 调整子图之间的间距
plt.tight_layout()

# 保存图像
plt.savefig(os.path.join(directory,'parameter_difference_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 创建实际参数值的分布图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
label_n = ['μs', 'Dc', 'tau']

for i in range(3):
    true_values = labels_tensor[:, i].numpy()
    
    # 绘制直方图，固定为15个直方格
    axes[i].hist(true_values, bins=15, color='skyblue', edgecolor='black')
    
    # 设置标题和标签
    axes[i].set_title(f'Distribution of {label_n[i]}')
    axes[i].set_xlabel(f'{label_n[i]} Value')
    axes[i].set_ylabel('Frequency')
    
    # 添加网格线
    axes[i].grid(True, linestyle='--', alpha=0.3)

# 调整子图之间的间距
plt.tight_layout()

# 保存图像
plt.savefig(os.path.join(directory,'parameter_distribution.png'), dpi=300, bbox_inches='tight')
plt.show()

# 在得到预测结果后，保存完整的标签和预测结果
predictions = predictions_tensor.numpy()

# 保存标签数据（包括断层位置）
with open(os.path.join(directory,'label.txt'), 'w') as f:
    for i in range(len(labels)):
        # 将前三个参数和断层位置信息组合成一行
        full_label = list(labels[i]) + list(fault_positions[i])  # 使用list()转换并连接
        # 将数字转换为字符串并用空格连接
        line = ' '.join(map(str, full_label))
        f.write(line + '\n')

# 保存预测数据（对于断层位置，我们使用真实值）
with open(os.path.join(directory,'prediction.txt'), 'w') as f:
    for i in range(len(predictions)):
        # 将预测的三个参数和真实的断层位置信息组合成一行
        full_prediction = list(predictions[i]) + list(fault_positions[i])  # 使用list()转换并连接
        # 将数字转换为字符串并用空格连接
        line = ' '.join(map(str, full_prediction))
        f.write(line + '\n')

print(f"Saved {len(labels)} records to label.txt and prediction.txt")
print("Each record contains 5 parameters: μs, Dc, tau, and two fault position parameters")

# 计算Dc参数的相对误差（使用绝对值）
dc_true = labels_tensor[:, 1].numpy()  # Dc是第二个参数
dc_pred = predictions_tensor[:, 1].numpy()
dc_relative_errors = np.abs((dc_pred - dc_true) / dc_true)

# 创建索引数组并根据相对误差排序
indices = np.arange(len(dc_relative_errors))
sorted_indices = indices[np.argsort(dc_relative_errors)]

# 选择10组差异大的（从最大误差开始）
high_error_indices = sorted_indices[-10:]
# 选择10组中等差异的（从中间开始）
mid_point = len(sorted_indices) // 2
medium_error_indices = sorted_indices[mid_point-5:mid_point+5]
# 选择10组差异小的（从最小误差开始）
low_error_indices = sorted_indices[:10]

# 保存选定的数据
with open(os.path.join(directory,'selected_cases.txt'), 'w') as f:
    # 写入表头
    f.write("=== High Error Cases (Relative Error > 10%) ===\n")
    f.write("True Values (μs, Dc, tau, fault_pos1, fault_pos2) | Predicted Values | Relative Error for Dc\n")
    
    # 保存高误差案例
    for idx in high_error_indices:
        true_vals = list(labels[idx]) + list(fault_positions[idx])
        pred_vals = list(predictions[idx]) + list(fault_positions[idx])
        rel_error = dc_relative_errors[idx] * 100  # 转换为百分比
        f.write(f"True: {' '.join(map(str, true_vals))} | ")
        f.write(f"Pred: {' '.join(map(str, pred_vals))} | ")
        f.write(f"Error: {rel_error:.2f}%\n")
    
    # 写入中等误差案例
    f.write("\n=== Medium Error Cases ===\n")
    f.write("True Values (μs, Dc, tau, fault_pos1, fault_pos2) | Predicted Values | Relative Error for Dc\n")
    for idx in medium_error_indices:
        true_vals = list(labels[idx]) + list(fault_positions[idx])
        pred_vals = list(predictions[idx]) + list(fault_positions[idx])
        rel_error = dc_relative_errors[idx] * 100
        f.write(f"True: {' '.join(map(str, true_vals))} | ")
        f.write(f"Pred: {' '.join(map(str, pred_vals))} | ")
        f.write(f"Error: {rel_error:.2f}%\n")
    
    # 写入低误差案例
    f.write("\n=== Low Error Cases ===\n")
    f.write("True Values (μs, Dc, tau, fault_pos1, fault_pos2) | Predicted Values | Relative Error for Dc\n")
    for idx in low_error_indices:
        true_vals = list(labels[idx]) + list(fault_positions[idx])
        pred_vals = list(predictions[idx]) + list(fault_positions[idx])
        rel_error = dc_relative_errors[idx] * 100
        f.write(f"True: {' '.join(map(str, true_vals))} | ")
        f.write(f"Pred: {' '.join(map(str, pred_vals))} | ")
        f.write(f"Error: {rel_error:.2f}%\n")

print("Selected cases have been saved to 'selected_cases.txt'")
print(f"High error range: {dc_relative_errors[high_error_indices[-1]]:.2f}% - {dc_relative_errors[high_error_indices[0]]:.2f}%")
print(f"Medium error range: {dc_relative_errors[medium_error_indices[0]]:.2f}% - {dc_relative_errors[medium_error_indices[-1]]:.2f}%")
print(f"Low error range: {dc_relative_errors[low_error_indices[0]]:.2f}% - {dc_relative_errors[low_error_indices[-1]]:.2f}%")
