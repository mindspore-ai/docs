# 快速入门

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/msadapter/docs/source_zh_cn/msadapter_user_guide/quick_start.md)

本文将为用户提供快速指引，以一个MNIST手写数字识别任务的完整流程为例，说明如何使用MSAdapter。并将一个完整的PyTorch代码用例适配至MSAdapter。若用户想直接运行MSAdapter的例子，可参考[MSAdapter适配后代码](#msadapter适配后代码)。

模型适配详细步骤如下：

1. 使能MSAdapter
2. 训练

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

### 1. 使能MSAdapter

MSAdapter已经兼容PyTorch的各类子模块，这里使能MSAdapter有两种方式第一种是微调脚本切换后端，第二种是环境变量切换。

#### 1.1 微调脚本切换后端

这种方式的前提是我们已经安装了MSAdapter，具体安装流程可以参考安装章节。需要在脚本前加一行，来切换后端，具体修改如下：

```python
import msadapter # 改为mindspore后端执行
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
```

#### 1.2 环境变量切换

这种方式要求我们下载源码，将源码路径加入到环境变量中来使能MSAdapter流程。

```bash
export PYTHONPATH=${work_dir}/msadapter/:$PYTHONPATH
export PYTHONPATH=${work_dir}/msadapter/msa_thirdparty:$PYTHONPATH
```

### 2. 训练

当前MSAdapter微分机制这部分已与torch对齐，无需用户去做调整。根据使能MSAdapter章节的修改即可拉起训练。

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
