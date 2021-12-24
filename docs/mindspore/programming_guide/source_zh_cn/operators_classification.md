# 算子分类

`Ascend` `GPU` `CPU` `入门`

<!-- TOC -->

- [算子分类](#算子分类)
    - [概述](#概述)
    - [Primitive算子](#primitive算子)
        - [计算算子](#计算算子)
            - [神经网络算子](#神经网络算子)
            - [数学算子](#数学算子)
            - [数组算子](#数组算子)
            - [通信算子](#通信算子)
        - [框架算子](#框架算子)
    - [nn算子](#nn算子)
        - [卷积层算子](#卷积层算子)
        - [池化层算子](#池化层算子)
        - [损失函数](#损失函数)
        - [优化器](#优化器)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_zh_cn/operators_classification.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## 概述

算子主要分为Primitivie算子和nn算子。所有的算子在Ascend AI处理器、GPU和CPU的支持情况，参见[算子支持列表](https://www.mindspore.cn/docs/note/zh-CN/master/operator_list.html)。

## Primitive算子

Primitive算子是开放给用户的最低阶算子接口，一个Primitive算子对应一个原语，它封装了底层的Ascend、GPU、AICPU、CPU等多种算子的具体实现，为用户提供基础算子能力。

Primitive算子接口是构建高阶接口、自动微分、网络模型等能力的基础。

Primitive算子可以分为[计算算子](#id3)和[框架算子](#id8)。计算算子主要负责具体的计算，而框架算子主要用于构图，自动微分等功能。

composite接口提供了一些预定义的组合算子，比如clip_by_value算子，以及涉及图变换的函数（GradOperation、Map）等，更多composite接口参见[composite接口](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.ops.html#composite)。

functional接口是为简化没有属性的Primitive算子调用流程而提供的，functional接口、composite接口和Primitive算子都可以从mindspore.ops中导入。

例如用户想使用pow功能，若使用Primitive算子，用户需要先实例化Pow算子，此时用户可以直接使用functional接口的tensor_pow来简化流程，代码示例如下：

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

input_x = mindspore.Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
input_y = 3.0
# 使用Primitive算子需先实例化
pow = ops.Pow()
output = pow(input_x, input_y)

# 直接使用functional接口
output = ops.tensor_pow(input_x, input_y)
print(output)
```

运行结果如下：

```text
[1. 8. 64.]
```

更多functional接口参见[functional接口](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.ops.html#functional)。

### 计算算子

计算算子按功能主要分为神经网络算子、数学算子、数组算子、通信算子等。

#### 神经网络算子

神经网络算子主要用于构建网络模型，比如卷积算子Conv2D，最大池化算子MaxPool等，参见[神经网络算子](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.ops.html#neural-network-operators)。

以下代码展示了最大池化算子MaxPool的使用：

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor(np.arange(1 * 3 * 3 * 4).reshape((1, 3, 3, 4)), mindspore.float32)
maxpool_op = ops.MaxPool(pad_mode="VALID", kernel_size=2, strides=1)
output = maxpool_op(x)
print(output)
```

运行结果如下：

```text
[[[[ 5.  6.  7.]
   [ 9. 10. 11.]]
  [[17. 18. 19.]
   [21. 22. 23.]]
  [[29. 30. 31.]
   [33. 34. 35.]]]]
```

#### 数学算子

数学算子主要是针对数学运算开发的算子，比如相加算子Add、求对数算子Log等，参见[数学算子](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.ops.html#math-operators)。

以下代码展示了求对数算子Log的使用：

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
log_oo = ops.Log()
output = log_oo(x)
print(output)
```

运行结果如下：

```text
[0.        0.6931472 1.3862944]
```

#### 数组算子

数组算子主要是针对数组类操作的算子，比如排序算子Sort、转置算子Transpose等，参见[数组算子](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.ops.html#array-operators)。

以下代码展示了转置算子Transpose的使用：

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
input_perm = (0, 2, 1)
transpose_op = ops.Transpose()
output = transpose_op(input_x, input_perm)
print(output)
```

运行结果如下：

```text
[[[ 1.  4.]
  [ 2.  5.]
  [ 3.  6.]]
 [[ 7. 10.]
  [ 8. 11.]
  [ 9. 12.]]]
```

#### 通信算子

通信算子主要是针对[多卡训练](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/distributed_training.html)时对各个卡进行通信的算子，比如收集算子AllGather、广播算子Broadcast等，参见[通信算子](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.ops.html#communication-operators)。

以下代码展示了收集算子AllGather的使用：

```python
# This example should be run with two devices. Refer to the tutorial > Distributed Training on mindspore.cn
import numpy as np
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.communication import init
from mindspore import Tensor, context

context.set_context(mode=context.GRAPH_MODE)
init()
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.allgather = ops.AllGather()
    def construct(self, x):
        return self.allgather(x)

input_x = Tensor(np.ones([2, 8]).astype(np.float32))
net = Net()
output = net(input_x)
print(output)
```

运行结果如下：

```text
[[1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 1. 1.]]
```

### 框架算子

`mindspore.ops.composite`中提供了一些涉及图变换的组合类算子，例如`MultitypeFuncGraph`、`HyperMap`和`GradOperation`等。

`MultitypeFuncGraph`用于定义一组重载的函数，用户可以使用该算子，根据不同类型，采用不同实现，参见[MultitypeFuncGraph](https://mindspore.cn/docs/programming_guide/zh-CN/master/hypermap.html#multitypefuncgraph)。

`HyperMap`可以对一组或多组输入做指定的运算，可以配合`MultitypeFuncGraph`一起使用，参见[HyperMap](https://mindspore.cn/docs/programming_guide/zh-CN/master/hypermap.html#hypermap)。

`GradOperation`用于生成输入函数的梯度，利用get_all、get_by_list和sens_param参数控制梯度的计算方式，参见[GradOperation](https://mindspore.cn/docs/programming_guide/zh-CN/master/grad_operation.html)。

## nn算子

nn算子是对低阶API的封装，主要包括卷积层算子、池化层算子、损失函数、优化器等。

nn算子还提供了部分与Primitive算子同名的接口，主要作用是对Primitive算子进行进一步封装，为用户提供更友好的API，当nn算子功能满足用户的要求时可以直接使用nn算子，而当nn算子功能无法满足用户特定要求时可以使用低阶的Primitive算子实现特定的功能。

### 卷积层算子

卷积层算子主要是在模型卷积层中使用的算子，比如卷积算子Conv2d、转置卷积算子Conv2dTranspose等，参见[卷积层算子](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.nn.html#convolution-layers)。

以下代码展示了卷积算子Conv2d的使用：

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

net = nn.Conv2d(120, 240, 4, has_bias=False, weight_init='normal')
x = Tensor(np.ones([1, 120, 1024, 640]), mindspore.float32)
output = net(x).shape
print(output)
```

运行结果如下：

```text
(1, 240, 1024, 640)
```

### 池化层算子

池化层算子主要是在模型池化层中使用的算子，比如平均池化算子AvgPool2d、最大池化算子MaxPool2d等，参见[池化层算子](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.nn.html#pooling-layers)。

以下代码展示了最大池化算子MaxPool2d的使用：

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

pool = nn.MaxPool2d(kernel_size=3, stride=1)
x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mindspore.float32)
output = pool(x)
print(output.shape)
```

运行结果如下：

```text
(1, 2, 2, 2)
```

### 损失函数

损失函数主要是用来评价模型的预测值和真实值的差异程度，常用的损失函数有BCEWithLogitsLoss、SoftmaxCrossEntropyWithLogits等，参见[损失函数](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.nn.html#loss-functions)。

以下代码展示了SoftmaxCrossEntropyWithLogits损失函数的使用：

```python
import numpy as np
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

loss = nn.SoftmaxCrossEntropyWithLogits()
logits = Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), mindspore.float32)
labels_np = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0]]).astype(np.float32)
labels = Tensor(labels_np)
output = loss(logits, labels)
print(output)
```

运行结果如下：

```text
[30.]
```

### 优化器

优化器主要是用于计算和更新梯度，常用的优化器有Adam、Momentum等，参见[优化器](https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.nn.html#optimizer-functions)。

以下代码展示了Momentum优化器的使用：

```python
import mindspore.nn as nn
from mindspore import Model

net = Net()
#1) All parameters use the same learning rate and weight decay
optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)

#2) Use parameter groups and set different values
conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
group_params = [{'params': conv_params, 'weight_decay': 0.01, 'grad_centralization':True},
                {'params': no_conv_params, 'lr': 0.01},
                {'order_params': net.trainable_params()}]
optim = nn.Momentum(group_params, learning_rate=0.1, momentum=0.9, weight_decay=0.0)
# The conv_params's parameters will use a learning rate of default value 0.1 and a weight decay of 0.01 and
# grad centralization of True.
# The no_conv_params's parameters will use a learning rate of 0.01 and a weight decay of default value 0.0
# and grad centralization of False..
# The final parameters order in which the optimizer will be followed is the value of 'order_params'.

loss = nn.SoftmaxCrossEntropyWithLogits()
model = Model(net, loss_fn=loss, optimizer=optim, metrics=None)
```
