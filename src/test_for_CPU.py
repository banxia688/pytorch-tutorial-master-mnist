import time
from numpy import argmax
from torch import no_grad
from torch.optim import Adam

from model import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


device = torch.device("cpu")  # 用 CPU 训练
print(device)


# 定义数据加载器函数
def get_data_loader(is_train):
    to_tensor = transforms.ToTensor()  # 创建 to_tensor 工具, 把图片类型转化成张量数据类型
    # 第一次跑程序时 dataset 自动下载到根目录 is_train 表示导入的是否是训练集
    dataset = MNIST("../", is_train, transform=to_tensor, download=True)
    return DataLoader(dataset, batch_size=128, shuffle=True)  # batch_size = 15 表示一个批次 15 张图片, 返回数据加载器, shuffle打乱训练数据


# 定义评估函数评估网络的识别正确率
def evaluate(test_data, net):
    correct = 0
    total = 0
    with no_grad():  # 计算精度过程不必计算梯度，节省计算资源
        for (img, target) in test_data:  # 从测试集中按批次取出数据
            outputs = net.forward(img.view(-1, 28 * 28))  # 正向传播
            for i, output in enumerate(outputs):  # 对批次中的每个结果进行比较
                if argmax(output) == target[i]:  # argmax 计算数列中一个最大值的序号, 对比神经网络识别的结果和正确结果, 累积正确个数
                    correct += 1
                total += 1
    return correct / total


def main():
    train_data = get_data_loader(is_train=True)  # 导入训练集
    test_data = get_data_loader(is_train=False)  # 导入测试集
    net = Net()  # 初始化神经网络

    # 设置超参数
    epoch = 2  # 训练轮次
    learning_rate = 1e-3  # 学习率 = 0.001
    # 设置优化方法———— Adaptive Momentum方法优化参数
    optimizer = Adam(net.parameters(), learning_rate)
    # 设置损失函数
    loss_function = torch.nn.NLLLoss()
    # 输出未经训练的神经网络的识别正确率, 十个数字随机取一个, 结果会接近 0.1
    print("Initial Accuracy: {}".format(evaluate(test_data, net)))

    # 开始训练
    net.train()  # Pytorch 框架固定写法，这里有没有都无所谓
    for i in range(epoch):
        start_time = time.time()  # 计时
        for (img, target) in train_data:  # 取出由 Dataloader 打包的训练数据
            net.zero_grad()  # 初始化梯度
            output = net.forward(img.view(-1, 28 * 28))  # 计算正向传播
            # 优化模型, 计算预测值 output 与 y 的差值
            loss = loss_function(output, target)  # nll_lost 为对数损失函数匹配前面的 LogSoftmax 中的对数运算
            loss.backward()  # 反向误差传播
            optimizer.step()  # 优化网络参数
        end_time = time.time()
        # 输出训练一轮所花费的时间
        print("Round {} elapsed time for CPU-based training: {}".format(i + 1, end_time - start_time))
        # 输出每训练一轮后的正确率
        print("Round {}".format(i + 1), "Accuracy: {}".format(evaluate(test_data, net)))

    output_image = 2  # 设置输出图像数量, 从 0 开始计数
    # 用测试集测试
    net.eval()  # Pytorch 框架固定写法，这里有没有都无所谓
    with torch.no_grad():  # 测试过程不必计算梯度，节省计算资源
        for (n, (img, _)) in enumerate(test_data):  # 取出由 Dataloader 打包的测试数据
            if n > output_image:
                break
            prediction = argmax(net.forward(img[0].view(-1, 28 * 28)))  # argmax选出预测概率最大的数字，即预测值
            plt.figure(n)  # 输出图像的数量
            plt.imshow(img[0].view(28, 28))  # 转成 28 * 28 * 1 的 Tensor 输出
            plt.title("Prediction: " + str(int(prediction)))   # 图像上方显示预测结果
        plt.show()  # 画出图像


if __name__ == "__main__":
    main()
