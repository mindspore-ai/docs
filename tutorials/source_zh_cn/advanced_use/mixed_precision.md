# 混合精度

## 概述

混合精度训练方法通过混合使用单精度和半精度数据格式来加速深度神经网络训练过程，同时保持了单精度训练所能达到的网络精度。混合精度训练能够加速计算过程，同时减少内存使用和存取，并在特定的硬件上可以训练更大的模型或batch size。

## 计算流程

MindSpore混合精度典型的计算流程如下图所示：

![mix precision](./images/mix_precision.jpg)

1. 参数以FP32存储；
2. 正向计算过程中，遇到FP16算子，需要把算子输入和参数从FP32 cast成FP16进行计算；
3. 将Loss层设置为FP32进行计算；
4. 反向计算过程中，首先乘以Loss Scale值，避免反向梯度过小而产生下溢；
5. FP16参数参与梯度计算，其结果将被cast回FP32；
6. 除以Loss scale值，还原被放大的梯度；
7. 判断梯度是否存在溢出，如果溢出则跳过更新，否则优化器以FP32对原始参数进行更新。

本文通过自动混合精度和手动混合精度的样例来讲解计算流程。

## 自动混合精度

使用自动混合精度，需要调用相应的接口，将待训练网络和优化器作为输入传进去；该接口会将整张网络的算子转换成FP16算子(除BatchNorm算子和Loss涉及到的算子外)。
另外要注意：使用混合精度后，一般要用上Loss Scale，避免数值计算溢出。

具体的实现步骤为：

1. 引入MindSpore的混合精度的接口amp；

2. 定义网络：该步骤和普通的网络定义没有区别(无需手动配置某个算子的精度)；

3. 使用amp.build_train_network()接口封装网络模型和优化器，在该步骤中MindSpore会将有需要的算子自动进行类型转换。

代码样例如下：

```python
# The interface of Auto_mixed precision
from mindspore.train import amp

# Define network
class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = P.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize network
net = LeNet5()

# Define training data, label and sens
predict = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
label = Tensor(np.zeros([1, 10]).astype(np.float32))
scaling_sens = Tensor(np.full((1), 1.0), dtype=mstype.float32)

# Define Loss and Optimizer
net.set_train()
loss = MSELoss()
optimizer = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
net_with_loss = WithLossCell(net, loss)
train_network = amp.build_train_network(net_with_loss, optimizer, level="O2")

# Run training
output = train_network(inputs, label, scaling_sens)
```

## 手动混合精度

MindSpore还支持手动混合精度。假定在网络中只有一个Dense Layer要用FP32计算，其他Layer都用FP16计算。混合精度配置以Cell为粒度，Cell默认是FP32类型。

以下是一个手动混合精度的实现步骤：

1. 定义网络: 该步骤与自动混合精度中的步骤2类似；注意：在LeNet中的fc3算子，需要手动配置成FP32；

2. 配置混合精度: LeNet通过net.add_flags_recursive(fp16=True)，把该Cell及其子Cell中所有的算子都配置成FP16；

3. 使用TrainOneStepWithLossScaleCell封装网络模型和优化器。

代码样例如下：

```python
# Define network
class LeNet5(nn.Cell):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, 10).add_flags_recursive(fp32=True)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = P.Flatten()

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize network and set mixing precision
net = LeNet5()
net.add_flags_recursive(fp16=True)

# Define training data, label and sens
predict = Tensor(np.ones([1, 1, 32, 32]).astype(np.float32) * 0.01)
label = Tensor(np.zeros([1, 10]).astype(np.float32))
scaling_sens = Tensor(np.full((1), 1.0), dtype=mstype.float32)

# Define Loss and Optimizer
net.set_train()
loss = MSELoss()
optimizer = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
net_with_loss = WithLossCell(net, loss)
train_network = TrainOneStepWithLossScaleCell(net_with_loss, optimizer)

# Run training
output = train_network(inputs, label, scaling_sens)
```
