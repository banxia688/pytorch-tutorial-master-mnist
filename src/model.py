import torch
from torch import nn
from torch.nn import Linear, Sequential, ReLU, LogSoftmax, MaxPool2d, Conv2d


# from mpl_toolkits.mplot3d import Axes3D
# 定义神经网络
class Net(nn.Module):

    def __init__(self):
        super().__init__()  # 继承父类

        # 使用 Sequential 定义四个全连接层
        self.model = Sequential(
            # 输入层把图像展成一维数组输入
            Linear(28 * 28, 64),
            # 选用整流函数为激活函数
            ReLU(),
            # 中间两层隐藏层各放置 64 个结点
            Linear(64, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            # 输出层共 10 个输出类别, 10 个结点
            Linear(64, 10),
            # 输出层通过 softmax 归一化, 并使用 LogSoftmax 对数运算提高运算稳定性
            LogSoftmax(dim=1)  # dim=1 表示列方向, 0 为行方向
        )

    # 定义前向传播过程
    def forward(self, x):
        x = self.model(x)
        return x


# 以下代码可以检验网络搭建得是否正确
""""""
model = Net()
print(model)
Input = torch.ones((64, 3, 1, 784))  # batch_size = 64, channel = 3, row = 1, columns = 784
Output = model(Input)
print(Output.shape)  # 如果网络参数正确, 输出torch.Size([(batch_size), 3, 1, 10])
"""
"""