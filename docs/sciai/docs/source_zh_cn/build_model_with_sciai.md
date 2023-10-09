# 使用SciAI构建神经网络

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/sciai/docs/source_zh_cn/build_model_with_sciai.md)&nbsp;&nbsp;

SciAI基础框架由若干基础模块构成，涵盖有神经网络搭建、训练、验证以及其他辅助函数等。

如下的示例展示了使用SciAI构建神经网络模型并进行训练的流程。

> 你可以在这里下载完整的样例代码：
> <https://gitee.com/mindspore/mindscience/tree/master/SciAI/tutorial>

## 模型构建基础

使用SciAI基础框架创建神经网络的原理与[使用MindSpore构建网络](https://www.mindspore.cn/tutorials/zh-CN/master/beginner/model.html)一致，但过程将会十分简便。

本章节以一个多层感知器为例，介绍了使用SciAI训练并求解如下方程。

$$
f(x) = {x_1}^2 + sin(x_2)
$$

该部分完整代码请参考[代码](https://gitee.com/mindspore/mindscience/blob/master/SciAI/tutorial/example_net.py)。

### 模型搭建

如下示例代码创建了一个输入维度为2，输出维度为1，包含两层维度为5的中间层的多层感知器。

```python
from sciai.architecture import MLP
from sciai.common.initializer import XavierTruncNormal

net = MLP(layers=[2, 5, 5, 1], weight_init=XavierTruncNormal(), bias_init='zeros', activation="tanh")
```

`MLP`将默认使用正态分布随机生成网络权重，偏差`bias`默认为0，激活函数默认为`tanh`。

`MLP`同时接受多样化的初始化方式和MindSpore提供的所有[激活函数](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.nn.html)，以及专为科学计算设计的激活函数。

### 损失函数定义

损失函数定义为[Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell)的子类， 并将损失的计算方法写在方法`construct`中。

```python
from mindspore import nn
from sciai.architecture import MSE

class ExampleLoss(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.mse = MSE()

    def construct(self, x, y_true):
        y_predict = self.network(x)
        return self.mse(y_predict - y_true)

net_loss = ExampleLoss(net)
```

此时，通过直接调用`net_loss`，并将输入`x`与真实值`y_true`作为参数，便可计算得到当前`net`预测的损失。

```python
from mindspore import Tensor

x = Tensor([[0.5, 0.5]])
y_true = Tensor([0.72942554])
print("loss value: ", net_loss(x, y_true))
# expected output
...
loss value: 0.3026065
```

### 模型训练与推理

得到损失函数后，我们即可使用SciAI框架中已封装好的训练类，使用数据集进行训练。
在本案例中，我们对方程进行随机采样，生成数据集`x_train`与`y_true`进行训练。

模型训练部分代码如下所示，其中主要展示了SciAI若干功能。
模型训练类`TrainCellWithCallBack`，其与[MindSpore.nn.TrainOneStepCell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.TrainOneStepCell.html#mindspore.nn.TrainOneStepCell)功能基本一致，
需要提供网络`net_loss`与优化器作为参数，并为科学计算功能增加了回调功能。
回调包括打印训练`loss`值、训练时间、自动保存`ckpt`文件。
如下的案例代码将会每100个训练周期打印`loss`值与训练时间，并在每1000个训练周期保存当前模型参数为`ckpt`文件。
SciAI提供`to_tensor`工具，可以方便地将多个`numpy`数据同时转换为`Tensor`类型。
使用`log_config`指定目标目录，用于自动保存`TrainCellWithCallBack`的回调打印，以及用户使用`print_log`所打印的内容。

```python
import numpy as np
from mindspore import nn
from sciai.common import TrainCellWithCallBack
from sciai.context import init_project
from sciai.utils import to_tensor, print_log, log_config

# Get the correct platform automatically and set to GRAPH_MODE by default.
init_project()
# Auto log saving
log_config("./logs")

def func(x):
    """The function to be learned to"""
    return x[:, 0:1] ** 2 + np.sin(x[:, 1:2])

optimizer = nn.Adam(net_loss.trainable_params())
trainer = TrainCellWithCallBack(net_loss, optimizer, loss_interval=100, time_interval=100, ckpt_interval=1000)
x_train = np.random.rand(1000, 2)
# Randomly collect ground truth
y_true = func(x_train)
# Convert to Tensor data type
x_train, y_true = to_tensor((x_train, y_true))
for _ in range(10001):
    trainer(x_train, y_true)
print_log("Finished")
```

预期运行结果如下。

```bash
python ./example_net.py
# expected output
...
step: 0, loss: 0.5189553, interval: 2.7039313316345215s, total: 2.7039313316345215s
step: 100, loss: 0.080132075, interval: 0.11984062194824219s, total: 2.8237719535827637s
step: 200, loss: 0.055663396, interval: 0.09104156494140625s, total: 2.91481351852417s
step: 300, loss: 0.032194577, interval: 0.09095025062561035s, total: 3.0057637691497803s
step: 400, loss: 0.015914217, interval: 0.09099435806274414s, total: 3.0967581272125244s
...
Finished
```

在训练结束并且损失收敛时，通过调用`y = net(x)`即可得到`x`处的预测值`y`。
继续随机采样若干位置`x_val`用于验证。

```python
x_val = np.random.rand(5, 2)
y_true = func(x_val)
y_pred = net(to_tensor(x_val)).asnumpy()
print_log("y_true:")
print_log(y_true)
print_log("y_pred:")
print_log(y_pred)
```

预期运行结果如下。经过训练，模型的预测值接近数值计算结果。

```bash
# expected output
y_true:
[[0.34606973]
 [0.70457536]
 [0.90531053]
 [0.84420218]
 [0.48239506]]
y_pred:
[[0.34271246]
 [0.70356864]
 [0.89893466]
 [0.8393946 ]
 [0.47805673]]
```

## 模型构建拓展

使用SciAI可以求解更为复杂的问题，例如物理驱动的神经网络（PINN）。该章节继续以一个多层感知器为例，介绍使用SciAI训练并求解如下偏微分方程。

$$
\frac{\partial{f}}{\partial{x}} - 2 \frac{f}{x} + {x}^2 {y}^2 = 0
$$

边界条件定义如下。

$$
f(0) = 0, f(1) = 1
$$

在此边界条件下，函数的解析解为：

$$
f(x) = \frac{x^2}{0.2 x^5 + 0.8}
$$

该部分完整代码请参考[代码](https://gitee.com/mindspore/mindscience/blob/master/SciAI/tutorial/example_grad_net.py)。

### 损失函数定义

与上一章中损失函数定义基本一致，需要定义损失为[Cell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell)的子类。

不同的是在该损失函数中，需要计算原函数的偏导。
SciAI为此提供了便捷的工具`operators.grad`，通过设置网络输入与输出的索引，可以计算某个输入对某个输出的偏导值。
在该问题中，输入输出维度均为1，因此设置`input_index`与`output_index`为0。

```python
from mindspore import nn, ops
from sciai.architecture import MSE, MLP
from sciai.operators import grad

class ExampleLoss(nn.Cell):
    """ Loss definition class"""
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.dy_dx = grad(net=self.network, output_index=0, input_index=0)  # partial differential definition
        self.mse = MSE()

    def construct(self, x, x_bc, y_bc_true):
        y = self.network(x)
        dy_dx = self.dy_dx(x)
        domain_res = dy_dx - 2 * ops.div(y, x) + ops.mul(ops.pow(x, 2), ops.pow(y, 2))  # PDE residual error

        y_bc = self.network(x_bc)
        bc_res = y_bc_true - y_bc  # Boundary conditions residual
        return self.mse(domain_res) + 10 * self.mse(bc_res)
```

### 模型训练与推理

通过终端执行脚本文件，执行训练与推理，得到如下预期结果。最终预测值`y_pred`与真实值`y_true`基本接近。

```bash
python ./example_grad_net.py
# expected output
...
step: 0, loss: 3.1961572, interval: 3.117840051651001s, total: 3.117840051651001s
step: 100, loss: 1.0862937, interval: 0.23533344268798828s, total: 3.353173494338989s
step: 200, loss: 0.7334847, interval: 0.21307134628295898s, total: 3.566244840621948s
step: 300, loss: 0.5629723, interval: 0.19696831703186035s, total: 3.763213157653809s
step: 400, loss: 0.4133342, interval: 0.20153212547302246s, total: 3.964745283126831s
...
Finished
y_true:
[[0.02245186]
 [0.99459697]
 [0.04027248]
 [0.12594332]
 [0.39779923]]
y_pred:
[[0.02293926]
 [0.99337316]
 [0.03924912]
 [0.12166673]
 [0.4006738 ]]
```