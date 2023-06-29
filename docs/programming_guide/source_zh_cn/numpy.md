# MindSpore NumPy函数使用介绍

<a href="https://gitee.com/mindspore/docs/blob/r1.2/docs/programming_guide/source_zh_cn/numpy.md" target="_blank"><img src="./_static/logo_source.png"></a>

## 概述

MindSpore NumPy工具包提供了一系列类NumPy接口。用户可以使用类NumPy语法在MindSpore上进行模型的搭建。

## 算子介绍

MindSpore Numpy具有四大功能模块：张量生成、张量操作、逻辑运算和其他常用数学运算。算子的具体相关信息可以参考[NumPy接口列表](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/mindspore.numpy.html)。

### 张量生成

生成类算子用来生成和构建具有指定数值、类型和形状的数组(Tensor)。

构建数组代码示例：

```python
import mindspore.numpy as np
import mindspore.ops as ops
input_x = np.array([1, 2, 3], np.float32)
print("input_x =", input_x)
print("type of input_x =", ops.typeof(input_x))
```

输出如下：

```python
input_x = [1. 2. 3.]
type of input_x = Tensor[Float32]
```

除了使用上述方法来创建外，也可以通过以下几种方式创建。

#### 生成具有相同元素的数组

生成具有相同元素的数组代码示例：

```python
import mindspore.numpy as np
input_x = np.full((2, 3), 6, np.float32)
print(input_x)
```

输出如下：

```python
[[6. 6. 6.]
 [6. 6. 6.]]
```

生成指定形状的全1数组，示例：

```python
import mindspore.numpy as np
input_x = np.ones((2, 3), np.float32)
print(input_x)
```

输出如下：

```python
[[1. 1. 1.]
 [1. 1. 1.]]
```

#### 生成具有某个范围内的数值的数组

生成指定范围内的等差数组代码示例：

```python
import mindspore.numpy as np
input_x = np.arange(0, 5, 1)
print(input_x)
```

输出如下：

```python
[0 1 2 3 4]
```

#### 生成特殊类型的数组

生成给定对角线处下方元素为1，上方元素为0的矩阵，示例：

```python
import mindspore.numpy as np
input_x = np.tri(3, 3, 1)
print(input_x)
```

输出如下：

```python
[[1. 1. 0.]
 [1. 1. 1.]
 [1. 1. 1.]]
```

生成对角线为1，其他元素为0的二维矩阵，示例：

```python
import mindspore.numpy as np
input_x = np.eye(2, 2)
print(input_x)
```

输出如下：

```python
[[1. 0.]
 [0. 1.]]
```

### 张量操作

变换类算子主要进行数组的维度变换，分割和拼接等。

#### 数组维度变换

矩阵转置，代码示例：

```python
import mindspore.numpy as np
input_x = np.arange(10).reshape(5, 2)
output = np.transpose(input_x)
print(output)
```

输出如下：

```python
[[0 2 4 6 8]
 [1 3 5 7 9]]
```

交换指定轴，代码示例：

```python
import mindspore.numpy as np
input_x = np.ones((1, 2, 3))
output = np.swapaxes(input_x, 0, 1)
print(output.shape)
```

输出如下：

```python
(2, 1, 3)
```

#### 数组分割

将输入数组平均切分为多个数组，代码示例：

```python
import mindspore.numpy as np
input_x = np.arange(9)
output = np.split(input_x, 3)
print(output)
```

输出如下：

```python
(Tensor(shape=[3], dtype=Int32, value= [0, 1, 2]),
 Tensor(shape=[3], dtype=Int32, value= [3, 4, 5]),
 Tensor(shape=[3], dtype=Int32, value= [6, 7, 8]))
```

#### 数组拼接

将两个数组按照指定轴进行拼接，代码示例：

```python
import mindspore.numpy as np
input_x = np.arange(0, 5)
input_y = np.arange(10, 15)
output = np.concatenate((input_x, input_y), axis=0)
print(output)
```

输出如下：

```python
[ 0  1  2  3  4 10 11 12 13 14]
```

### 逻辑运算

逻辑计算类算子主要进行逻辑运算。

相等（equal）和小于（less）计算代码示例如下：

```python
import mindspore.numpy as np
input_x = np.arange(0, 5)
input_y = np.arange(0, 10, 2)
output = np.equal(input_x, input_y)
print("output of equal:", output)
output = np.less(input_x, input_y)
print("output of less:", output)
```

输出如下：

```python
output of equal: [ True False False False False]
output of less: [False  True  True  True  True]
```

### 数学运算

数学计算类算子主要进行各类数学计算：
加减乘除乘方，以及指数、对数等常见函数等。

数学计算支持类似NumPy的广播特性。

#### 加法

以下代码实现了`input_x`和`input_y`两数组相加的操作：

```python
import mindspore.numpy as np
input_x = np.full((3, 2), [1, 2])
input_y = np.full((3, 2), [3, 4])
output = np.add(input_x, input_y)
print(output)
```

输出如下：

```python
[[4 6]
 [4 6]
 [4 6]]
```

#### 矩阵乘法

以下代码实现了`input_x`和`input_y`两矩阵相乘的操作：

```python
import mindspore.numpy as np
input_x = np.arange(2*3).reshape(2, 3).astype('float32')
input_y = np.arange(3*4).reshape(3, 4).astype('float32')
output = np.matmul(input_x, input_y)
print(output)
```

输出如下：

```python
[[20. 23. 26. 29.]
 [56. 68. 80. 92.]]
```

#### 求平均值

以下代码实现了求`input_x`所有元素的平均值的操作：

```python
import mindspore.numpy as np
input_x = np.arange(6).astype('float32')
output = np.mean(input_x)
print(output)
```

输出如下：

```python
2.5
```

#### 指数

以下代码实现了自然常数`e`的`input_x`次方的操作：

```python
import mindspore.numpy as np
input_x = np.arange(5).astype('float32')
output = np.exp(input_x)
print(output)
```

输出如下：

```python
[ 1.         2.718282   7.3890557 20.085537  54.598145 ]
```

## MindSpore Numpy与MindSpore特性结合

`mindspore.numpy`能够充分利用MindSpore的强大功能，实现算子的自动微分，并使用图模式加速运算，帮助用户快速构建高效的模型。同时，MindSpore还支持多种后端设备，包括`Ascend`、`GPU`和`CPU`等，用户可以根据自己的需求灵活设置。以下提供了几种常用方法：

- `ms_function`: 将代码包裹进图模式，用于提高代码运行效率。
- `GradOperation`: 用于自动求导。
- `mindspore.context`: 用于设置运行模式和后端设备等。
- `mindspore.nn.Cell`: 用于建立深度学习模型。

### ms_function使用示例

首先，以神经网络里经常使用到的矩阵乘与矩阵加算子为例：

```python
import mindspore.numpy as np

x = np.arange(8).reshape(2, 4).astype('float32')
w1 = np.ones((4, 8))
b1 = np.zeros((8,))
w2 = np.ones((8, 16))
b2 = np.zeros((16,))
w3 = np.ones((16, 4))
b3 = np.zeros((4,))

def forward(x, w1, b1, w2, b2, w3, b3):
    x = np.dot(x, w1) + b1
    x = np.dot(x, w2) + b2
    x = np.dot(x, w3) + b3
    return x

print(forward(x, w1, b1, w2, b2, w3, b3))
```

输出如下:

```python
[[ 768.  768.  768.  768.]
 [2816. 2816. 2816. 2816.]]
```

对上述示例，我们可以借助`ms_function`将所有算子编译到一张静态图里以加快运行效率，示例如下：

```python
from mindspore import ms_function

forward_compiled = ms_function(forward)
```

> 目前静态图不支持在命令行模式中运行，并且有部分语法限制。`ms_function`的更多信息可参考[API: ms_function](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/mindspore.html#mindspore.ms_function)。

### GradOperation使用示例

`GradOperation` 可以实现自动求导。以下示例可以实现对上述没有用`ms_function`修饰的`forward`函数定义的计算求导。

```python
from mindspore import ops

grad_all = ops.composite.GradOperation(get_all=True)
grad_all(forward)(x, w1, b1, w2, b2, w3, b3)
```

如果要对`ms_function`修饰的`forward`计算求导，需要提前使用`context`设置运算模式为图模式，示例如下：

```python
from mindspore import ops, ms_function, context

context.set_context(mode=context.GRAPH_MODE)

grad_all = ops.composite.GradOperation(get_all=True)
grad_all(ms_function(forward))(x, w1, b1, w2, b2, w3, b3)
```

 更多细节可参考[API: GradOperation](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/ops/mindspore.ops.GradOperation.html)。

### mindspore.context使用示例

MindSpore支持多后端运算，可以通过`mindspore.context`进行设置。`mindspore.numpy` 的多数算子可以使用图模式或者PyNative模式运行，也可以运行在CPU，CPU或者Ascend等多种后端设备上。

```python
from mindspore import context

# Execucation in static graph mode
context.set_context(mode=context.GRAPH_MODE)

# Execucation in PyNative mode
context.set_context(mode=context.PYNATIVE_MODE)

# Execucation on CPU backend
context.set_context(device_target="CPU")

# Execucation on GPU backend
context.set_context(device_target="GPU")

# Execucation on Ascend backend
context.set_context(device_target="Ascend")
...
```

 更多细节可参考[API: mindspore.context](https://www.mindspore.cn/doc/api_python/zh-CN/r1.2/mindspore/mindspore.context.html)。

### mindspore.numpy使用示例

这里提供一个使用`mindspore.numpy`构建网络模型的示例。

`mindspore.numpy` 接口可以定义在`nn.Cell`代码块内进行网络的构建，示例如下：

```python
import mindspore.numpy as np
from mindspore import context
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)

x = np.arange(8).reshape(2, 4).astype('float32')
w1 = np.ones((4, 8))
b1 = np.zeros((8,))
w2 = np.ones((8, 16))
b2 = np.zeros((16,))
w3 = np.ones((16, 4))
b3 = np.zeros((4,))

class NeuralNetwork(Cell):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

    def construct(self, x, w1, b1, w2, b2, w3, b3):
        x = np.dot(x, w1) + b1
        x = np.dot(x, w2) + b2
        x = np.dot(x, w3) + b3
        return x

net = NeuralNetwork()

print(net(x, w1, b1, w2, b2, w3, b3))
```

输出如下:

```python
[[ 768.  768.  768.  768.]
 [2816. 2816. 2816. 2816.]]
```

更多构建网络的细节可以参考[MindSpore训练指导](https://www.mindspore.cn/tutorial/training/zh-CN/r1.2/index.html)。
