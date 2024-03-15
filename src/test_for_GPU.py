import time
import torch.nn

from model import *
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# cmd 中输入 nvidia-smi, 表格中第一列显示 cuda: 0 所对应的你电脑的显卡
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 有 GPU 则用 GPU 无则用 CPU
# print(“Training on device: {}.”.format(device))


# 定义数据加载器函数
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])  # 定义 to_tensor 数据转换类型
    # 第一次跑程序时 dataset 自动下载到根目录 is_train 表示导入的是否是训练集
    dataset = MNIST("../", is_train, transform=to_tensor, download=True)
    return DataLoader(dataset, batch_size=64, shuffle=True)  # batch_size = 15 表示一个批次 15 张图片, shuffle 打乱训练数据


# 定义评估函数评估神经网络的识别正确率
def evaluate(test_data, net):
    correct = 0
    total = 0
    with torch.no_grad():  # 评估时不必计算梯度
        for x, y in test_data:  # 从测试集中按批次取出数据
            x = x.to(device)
            y = y.to(device)
            outputs = net(x.view(-1, 28 * 28))  # 计算神经网络的预测值
            for i, output in enumerate(outputs):  # 对批次中的每个结果进行比较
                if torch.argmax(output) == y[i]:  # argmax 计算数列中一个最大值的序号, 对比神经网络识别的结果和正确结果, 累积正确个数
                    correct += 1
                total += 1
    return correct / total


def main():
    # 设置超参数
    epoch = 5  # 训练轮次
    learning_rate = 1e-3  # 学习率 = 0.001

    train_data = get_data_loader(is_train=True)  # 导入训练集
    test_data = get_data_loader(is_train=False)  # 导入测试集
    net = Net()  # 初始化神经网络
    net.to(device)  # 使用 GPU 初始化模型, 无需重新赋值 <-------------------------------------->

    # 先输出未经训练的神经网络的识别正确率, 十个数字随机取一个, 结果会接近 0.1
    print("Initial Accuracy: {}".format(evaluate(test_data, net)))
    # 设置损失函数
    loss_function = torch.nn.NLLLoss()
    loss_function.to(device)  # 使用 GPU 计算损失, 无需重新赋值 <------------------------------->
    # 设置优化方法———— Adaptive Momentum 方法优化参数
    optimizer = torch.optim.Adam(net.parameters(), learning_rate)

    # 把网络切换成训练模式并开始训练
    net.train()  # Pytorch 框架固定写法, 这里有没有都无所谓, 写上是一个好习惯, 将来有 dropout 层时就有用
    for i in range(1, epoch + 1):  # 轮次从 1 开始更自然
        loss_total = 0.0
        start_time = time.time()  # 计时
        for x, y in train_data:  # 取出训练数据 x 为输入, y 为正确标签
            # variable.to(device)使用 GPU 训练, 需要对数据重新赋值 <------------------------------------->
            x = x.to(device)
            y = y.to(device)
            # 计算正向传播, -1 表示自动计算 batch_size
            output = net(x.view(-1, 28 * 28))
            # 计算损失（属于正向传播的一部分）
            loss = loss_function(output, y)  # 计算预测值 output 与 y 的差值, nll_loss 为对数损失函数, 匹配 LogSoftmax 的对数运算
            # 对上一训练轮次的梯度初始化
            net.zero_grad()
            # 反向误差传播
            loss.backward()
            # 优化网络参数
            optimizer.step()
            loss_total += loss

        end_time = time.time()
        # 输出训练一轮所花费的时间
        print("Round {} elapsed time for GPU-based training: {}".format(i, end_time - start_time))
        # 输出每训练一轮后的正确率
        print("Round {}".format(i), "Accuracy: {}".format(evaluate(test_data, net)))
        if i == 1 or i % 10 == 0:  # 每十轮打印一次损失函数值
            # 输出一轮训练中计算的误差值
            print("Round {}".format(i), "loss_total: {}".format(loss_total / len(train_data)))

    output_image = 2  # 设置输出图像数量, 从 0 开始计数
    # 把网络切换成评估模式并用测试集测试
    net.eval()  # Pytorch 框架固定写法，这里有没有都无所谓
    with torch.no_grad():  # 测试过程不必计算梯度，节省计算资源
        for (n, (img, _)) in enumerate(test_data):  # 枚举函数, 允许直接取出一组数据中的索引值和实际值
            if n > output_image:
                break
            img = img.to(device)
            prediction = torch.argmax(net.forward(img[0].view(-1, 28 * 28)))  # argmax 选出预测概率最大的数字, 即预测值
            prediction = prediction.to(device)
            plt.figure(n)  # 输出图像的数量
            # Tensor.cpu() 先把张量转成转移回 cpu中, 再用 view 来改变张量的形状 (28 * 28 * (1)) 作为 imshow 函数的输入
            plt.imshow(Tensor.cpu(img[0]).view(28, 28))
            plt.title("Prediction: " + str(int(prediction)))  # 图像上方显示预测结果
        plt.show()  # 画出图像


if __name__ == "__main__":
    main()
