# 快速入门

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/msadapter/docs/source_zh_cn/msadapter_user_guide/quick_start.md)

本文将为用户提供快速指引，以一个MNIST手写数字识别任务的完整流程为例，说明如何使用MSAdapter。并将一个完整的PyTorch代码用例适配至MSAdapter。若用户想直接运行MSAdapter的例子，可参考[MSAdapter适配后代码](#msadapter适配后代码)。

模型适配详细步骤如下：

1. 导入依赖包
2. 模型定义
3. 参数解析
4. 数据下载与预处理
5. 模型构建
6. 定义损失函数
7. 训练

## PyTorch用例

以下为一个基础MNIST手写数字识别的PyTorch用例，步骤与上述相同。
代码使用的是CUDA版本，如果想使用CPU版本删除代码中的`.to('cuda')`内容。

```python
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def parse_args():
    parser = argparse.ArgumentParser(description="command line arguments")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    return parser.parse_args()

def data_process(inputs, labels):
    inputs = inputs.view(inputs.size(0), -1)
    return inputs, labels

def main():
    # 获取传参
    args = parse_args()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 加载数据集
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model = ToyModel().to('cuda')
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    step = 0

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = data_process(inputs, labels)
            inputs, labels = inputs.to('cuda'), labels.to('cuda')  
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels).to('cuda')
            loss.backward()
            optimizer.step()
            # 添加每个step的打印，用户可自行修改
            print(f"step = {step}, loss : {loss}")
            step += 1

if __name__ == "__main__":
    main()
```

## MSAdapter详细适配步骤

接下来，对应PyTorch的完整流程，说明如何使用MSAdapter完成相同的任务。

### 1. 导入依赖包

MSAdapter已经兼容PyTorch的各类子模块，无需修改。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

### 2. 模型定义

MSAdapter已经兼容torch.nn.Module，无需修改。

```python
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(64, 10)
    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
```

### 3. 参数解析

argparse是常规Python，与深度学习无关，无需修改。

### 4. 数据下载与预处理

MSAdapter已经兼容基础数据集相关接口，无需修改。

```python
def data_process(inputs, labels):
    inputs = inputs.view(inputs.size(0), -1)
    return inputs, labels

# 预处理函数
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 加载数据集
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
```

### 5. 模型构建

MSAdapter在torch.nn.Module.to()调用上与PyTorch有差别。

```python
model = ToyModel().to('cuda')
```

由于MSAdapter暂时不支持torch.nn.Module.to接口，需要转换为如下方式，MSAdapter默认将模型放置于NPU上。若用户希望将模型或者张量搬运至CPU，则需要调用.cpu()接口。

修改如下：

```python
model = ToyModel()
```

### 6. 定义损失函数

MSAdapter的损失函数使用方式与PyTorch一致，无需修改。

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
```

### 7. 训练

MSAdapter在torch.Tensor.to()、正向计算、反向微分计算的调用上与PyTorch有差别，需要修改代码。

```python
for epoch in range(args.epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = data_process(inputs, labels)
        inputs, labels = inputs.to('cuda'), labels.to('cuda')  # Tensor.to()问题
        optimizer.zero_grad()
        outputs = model(inputs) # 前向调用不同
        loss = criterion(outputs, labels).to('cuda') # Tensor.to()问题
        loss.backward() # 反向调用不同
        optimizer.step()
        step += 1
```

#### torch.Tensor.to()

与步骤5类似，由于MSAdapter暂时不支持torch.nn.Tensor.to接口，需要转换为如下方式。注意：MSAdapter默认将模型放置于NPU上。

```python
inputs, labels = inputs.to('cuda'), labels.to('cuda')
loss = criterion(outputs, labels).to('cuda')
```

修改如下：

```python
# inputs, labels无需显示指定NPU
loss = criterion(outputs, labels)
```

#### 前向与反向计算

由于MSAdapter使用了函数式微分，正向反向计算均要调用函数，所以需要将PyTorch模型封装为函数。用户除了修改代码，还需要导入MindSpore。

```python
outputs = model(inputs) # 前向调用不同
loss = criterion(outputs, labels).to('cuda') # Tensor.to()问题不在此重述
loss.backward() # 反向调用不同
```

修改如下：

1. 预定义正向计算函数和反向计算函数

    ```python
    import mindspore
    def forward_fn(inputs, labels):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        return loss
    grad_fn = mindspore.value_and_grad(forward_fn, None, weights=model.trainable_params())
    ```

2. 替换原始的PyTorch计算过程

    ```python
    loss, grads = grad_fn(inputs, labels)
    ```

## MSAdapter适配后代码

此处提供MSAdapter可运行的代码：

```python
import argparse
import torch
import torch_npu
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mindspore

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

def parse_args():
    parser = argparse.ArgumentParser(description="command line arguments")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    return parser.parse_args()

def data_process(inputs, labels):
    inputs = inputs.view(inputs.size(0), -1)
    return inputs, labels

def main():
    # 获取传参
    args = parse_args()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 加载数据集
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # 将模型转移到NPU上
    model = ToyModel()
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    step = 0

    def forward_fn(inputs, labels):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        return loss
    grad_fn = mindspore.value_and_grad(forward_fn, None, weights=model.trainable_params())

    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            # 数据预处理，将数据集的数据转成需要的shape
            inputs, labels = data_process(inputs, labels)

            optimizer.zero_grad()
            loss, grads = grad_fn(inputs, labels)
            optimizer.step()

            # 添加每个step的打印，用户可自行修改
            print(f"step = {step}, loss : {loss}")
            step += 1

if __name__ == "__main__":
    main()
```

## loss对比

由于硬件不同的原因，两者实际的运行结果（如模型参数、loss等）会有出入。

epoch=1时，一共937个step，loss如下：

**PyTorch loss**

```text
step = 930, loss : 0.37795058
step = 931, loss : 0.48661083
step = 932, loss : 0.46579897
step = 933, loss : 0.54568535
step = 934, loss : 0.46733740
step = 935, loss : 0.32921690
step = 936, loss : 0.37337211
step = 937, loss : 0.31820250
```

**MSAdapter loss**

```text
step = 930, loss : 0.42702404
step = 931, loss : 0.55013794
step = 932, loss : 0.37097090
step = 933, loss : 0.36169168
step = 934, loss : 0.57616550
step = 935, loss : 0.37290677
step = 936, loss : 0.52857995
step = 937, loss : 0.51202524
```
