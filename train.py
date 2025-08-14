import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from SeismicNet import SeismicNet
import torch
import time
from sklearn.model_selection import KFold

print(torch.__version__)
print(torch.cuda.current_device())  # 显示当前 CUDA 设备的索引
print(torch.cuda.get_device_name(0))  # 显示特定 CUDA 设备的名称

# torch.cuda.set_device(0)  # 指定使用编号为0的GPU
LEARNING_RATE = 0.0001
# 结果保存文件夹

directory = '/Nas/Liqy/EarthquakeSeismicNet/MR/result250728/'

# model_name 决定模型最终的name
model_name = '250728_1.pth'

# 是否加载之前的模型参数
load_previous_params = 0  # 根据需要设置为True或False
lr_maxs = LEARNING_RATE  # 初始学习率
lr_mins = 0.00001  # LEARNING_RATE / 10  # 最小学习率
T_max = 500
T_cycles = 500
num_epochs = T_max
Batch = 128
label_counts = 3

def normalize_data(data, min_val, max_val):
    """
    对给定的数据进行最小-最大归一化。
    """
    # 避免除以零的情况，当最小值和最大值相等时
    if max_val - min_val == 0:
        return data
    return (data - min_val) / (max_val - min_val)

def find_labels_ranges(label, label_count=label_counts):
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

def denormalize_label_predictions(pre, min_vals, max_vals, label_count=label_counts):
    """
    对批量预测结果的每个标签分别进行反归一化。

    参数:
    - pre: 预测结果，为一个包含多个批次预测的列表，每个批次预测为一个32x3的tensor。
    - min_vals: 每个标签的最小值列表。
    - max_vals: 每个标签的最大值列表。

    返回:
    - denormalized_predictions: 反归一化后的预测结果列表。
    """
    denormalized_predictions = []
    for pred in pre:
        # pred现在是32x3的tensor，直接对每个标签进行反归一化
        for i in range(label_count):  # 对3个标签分别进行处理
            pred[i] = (pred[i] * (max_vals[i] - min_vals[i])) + min_vals[i]
        denormalized_predictions.append(pred)

    # 由于predictions已经是一个列表，直接返回处理后的列表即可
    return denormalized_predictions

def load_data(input_fold, label_fold, target_length=3001, label_count = label_counts):

    # 这里同时对数据和标签进行归一化处理，精细化处理了。注意对于数据而言，其最小值为0

    input_data = []
    label = []

    # 扫描数据以找到最小值和最大值

    min_val, max_val= find_data_range(input_fold)
    labels_min_vals, labels_max_vals = find_labels_ranges(label_fold)
    entries = os.listdir(input_fold)
    for entry in entries:
        if os.path.isfile(os.path.join(input_fold, entry)):
            input_file_path = os.path.join(input_fold, entry)
            with open(input_file_path, 'r') as file:
                data = file.readline().strip().split()
                data = [float(x) for x in data]
                
                data = [min_val] + data

                # max_data = max(data)
                # threshold = max_data / 400
                # zero_idx = None
                # for i, val in enumerate(data[1:]):
                #     if abs(val) < threshold:
                #         zero_idx = i
                #         break
                # if zero_idx is not None:
                #     data[zero_idx:] = [0.0] * len(data[zero_idx:])

                if len(data) < target_length:
                    data += [min_val] * (target_length - len(data))
                data = normalize_data(np.array(data), min_val, max_val)
                input_data.append(data)

    # 加载和归一化标签数据
    entries = os.listdir(label_fold)
    for entry in entries:
        if os.path.isfile(os.path.join(label_fold, entry)):
            label_file_path = os.path.join(label_fold, entry)
            with open(label_file_path, 'r') as file:
                file.readline()  # 跳过第一行
                labels_data = file.readline().strip().split()
                labels_data = [float(x) for x in labels_data]
                # 分别归一化每个标签
                normalized_labels = [normalize_data(np.array(labels_data[j]), labels_min_vals[j], labels_max_vals[j]) for j
                                    in range(label_count)]
                label.append(normalized_labels)

    input_data = np.array(input_data)
    label = np.array(label).reshape(-1, label_count)  # 重新整理形状以匹配预期的标签形状

    return torch.tensor(input_data, dtype=torch.float32), torch.tensor(label,
                                                                       dtype=torch.float32), labels_min_vals, labels_max_vals

# 加载数据
input_folder = r'/Nas/Liqy/EarthquakeSeismicNet/data_only_90/MR'
label_folder = r'/Nas/Liqy/EarthquakeSeismicNet/data_only_90/parameters'

inputs, labels, labels_min_val, labels_max_val = load_data(input_folder, label_folder)

# 划分数据集
train_inputs, val_inputs, train_labels, val_labels = train_test_split(
    inputs, labels, test_size=0.11, random_state=42
)

# 创建 TensorDataset和 DataLoader
train_dataset = TensorDataset(train_inputs, train_labels)
val_dataset = TensorDataset(val_inputs, val_labels)

train_loader = DataLoader(train_dataset, batch_size=Batch, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Batch, shuffle=False)

def set_seed(seed):
    """固定随机数种子以确保结果可复现。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用CUDA
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机数种子
set_seed(30)

def cosine_annealing_schedule_mod(t, lr_max, lr_min, T_cycle):
    """
    计算在第t个epoch时的学习率，考虑周期性变化。

    参数:
    - t: 当前epoch。
    - lr_max: 最大学习率。
    - lr_min: 最小学习率。
    - T_max: 总epoch数。
    - T_cycle: 每个学习率周期的epoch数。

    返回:
    - 当前epoch的学习率。
    """
    t = t % T_cycle
    lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t / T_cycle))
    return lr_t

# 更新优化器的学习率
def update_learning_rate(opt, l_r):
    for param_group in opt.param_groups:
        param_group['lr'] = l_r

# 检查是否有可用的GPU，如果有，使用第一个GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义模型
model = SeismicNet().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# optimizer = optim.Adam(model.parameters(), lr=0.001)
# scheduler = StepLR(optimizer, step_size=300, gamma=0.1)

# 计算每个epoch的学习率
lr_each_epoch = [cosine_annealing_schedule_mod(t, lr_maxs, lr_mins, T_cycles) for t in range(T_max)]

# 使用先前的模型
if load_previous_params:
    model_path = directory + '24115_1_500.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # 如果GPU可用，则加载模型和优化器到 GPU
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 指定使用编号为0的GPU
            model.cuda()
            print("Loaded previous model parameters to GPU.")
        else:
            print("Loaded previous model parameters to CPU.")

        # 加载优化器状态
        if 'optimizer_state_dict' in checkpoint:
            print("加载优化器")
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state from checkpoint.")
        else:
            print("No optimizer state found in checkpoint; starting with a fresh optimizer.")
    else:
        print("Previous model parameters not found. Starting training from scratch.")
else:
    print("Starting training from scratch.")

# 准备记录损失的列表
train_losses = []
val_losses = []
val_labels1 = []
val_predictions1 = []
train_predictions1 = []
train_labels1 = []

# 训练和验证循环
for epoch in range(num_epochs):
    epoch_start_time = time.time() #记录epoch开始时间
    lr = lr_each_epoch[epoch]  # 从之前计算的列表中获取当前epoch的学习率
    update_learning_rate(optimizer, lr)  # 更新学习率

    model.train()
    train_loss = 0
    for batch_inputs, batch_labels in train_loader:
        batch_inputs = batch_inputs.unsqueeze(1)
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        predictions = model(batch_inputs)
        loss = loss_fn(predictions, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # 添加预测和标签用于分析
        train_predictions1.append(predictions.detach())  # 确保从图中脱离
        train_labels1.append(batch_labels.detach())
    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            batch_inputs = batch_inputs.unsqueeze(1)

            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_labels)
            val_loss += loss.item()
            # 添加验证预测和标签
            val_predictions1.append(predictions.detach())  # 从图中脱离
            val_labels1.append(batch_labels.detach())
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    epoch_time = time.time() - epoch_start_time  # 计算epoch用时
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}, learning rate: {current_lr}，time：{epoch_time}')

# 处理文件名
base_name = model_name[:-4]  # 移除 ".pth" 后缀
filename = f"{base_name}_{T_max}.pth"
save_path = f'{directory}{filename}'
# 保存模型参数
# torch.save(model.state_dict(), save_path)
# 保存模型和优化器
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, save_path)

# 绘制训练和验证损失图
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(directory + model_name[:-4] + str(T_max) + '.png')  # 保存损失曲线图
plt.show()

# 打开一个文件用于写入
txt_name = directory + model_name[:-4] + str(T_max) + '.txt'
with open(txt_name, 'w') as f:
    # 同时遍历两个列表，并将它们的元素写入文件的同一行，用逗号分隔
    for item1, item2 in zip(train_losses, val_losses):
        f.write(f"{item1}, {item2}\n")


# val_predictions_denormalized = [denormalize_label_predictions(pred, labels_min_val, labels_max_val) for pred in
# val_predictions1]
# val_label_denormalized = [denormalize_label_predictions(label, labels_min_val, labels_max_val) for label in
# val_labels1]
# 保存所有epoch训练集和验证集预测结果
# 注意：这里简单地保存为Tensor，具体格式可能需要根据实际需要调整

torch.save(val_predictions1, directory + base_name + 'val_predictions' + str(T_max) + '.pt')
torch.save(val_labels1, directory + base_name + 'val_label' + str(T_max) + '.pt')
print(f'labels_min_val: {labels_min_val}, labels_max_val: {labels_max_val}') 