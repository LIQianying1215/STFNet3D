import torch.nn as nn
import torch.nn.functional as F


class SeismicNet(nn.Module):
    def __init__(self):
        super(SeismicNet, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1)

        # 根据卷积后的输出尺寸确定线性层的输入尺寸
        # 这里我们需要计算经过5次卷积后的尺寸
        self.fc_input_size = self._get_conv_output_size(3001)

        # 定义全连接层
        self.fc = nn.Linear(self.fc_input_size, 3)

    def _get_conv_output_size(self, size):
        # 通过卷积层和池化层计算最终的输出尺寸
        size = self._conv_size(size, 7, 2, 3)
        size = self._conv_size(size, 7, 2, 3)
        size = self._conv_size(size, 7, 2, 3)
        size = self._conv_size(size, 4, 2, 1)
        size = self._conv_size(size, 4, 2, 1)
        return size * 256  # 乘以最后一个卷积层的输出通道数

    def _conv_size(self, size, kernel_size, stride, padding):
        return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        # 通过卷积层和激活层
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # 准备输入全连接层
        x = x.view(-1, self.fc_input_size)

        # 全连接层
        x = self.fc(x)
        return x


if __name__ == "__main__":

    loss_fn = nn.MSELoss()
    model = SeismicNet()

    print(model)