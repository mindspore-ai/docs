# 求导

`Ascend` `GPU` `CPU` `模型开发`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_zh_cn/grad_operation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

## 概述

MindSpore的GradOperation接口用于生成输入函数的梯度，利用get_all、get_by_list和sens_param参数控制梯度的计算方式。

本文主要介绍如何进行一阶、二阶求导，以及如何停止计算梯度。更多内容可参见[API文档](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.GradOperation.html) 。

## 一阶求导

MindSpore计算一阶导数方法`mindspore.ops.GradOperation (get_all=False, get_by_list=False, sens_param=False)`，其中`get_all`为`False`时，只会对第一个输入求导，为`True`时，会对所有输入求导；`get_by_list`为`False`时，不会对权重求导，为`True`时，会对权重求导；`sens_param`对网络的输出值做缩放以改变最终梯度，故其维度与输出维度保持一致。下面用MatMul算子的一阶求导做深入分析。

完整样例代码见：[一阶求导样例代码](https://gitee.com/mindspore/docs/tree/r1.6/docs/sample_code/high_order_differentiation/first_order)

### 输入求导

对输入求导代码如下：

```python
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import ParameterTuple, Parameter
from mindspore import dtype as mstype
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')
    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out

class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()
    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
output = GradNetWrtX(Net())(x, y)
print(output)
```

输出结果如下：

```text
[[4.5099998 2.7 3.6000001]
 [4.5099998 2.7 3.6000001]]
```

为便于分析，输入`x`、`y`以及权重`z`可以表示成如下形式:

```python
x = Tensor([[x1, x2, x3], [x4, x5, x6]])  
y = Tensor([[y1, y2, y3], [y4, y5, y6], [y7, y8, y9]])
z = Tensor([z])
```

根据MatMul算子定义可得前向结果：

$output = [[(x1 \cdot y1 + x2 \cdot y4 + x3 \cdot y7) \cdot z, (x1 \cdot y2 + x2 \cdot y5 + x3 \cdot y8) \cdot z, (x1 \cdot y3 + x2 \cdot y6 + x3 \cdot y9) \cdot z]$,

$[(x4 \cdot y1 + x5 \cdot y4 + x6 \cdot y7) \cdot z, (x4 \cdot y2 + x5 \cdot y5 + x6 \cdot y8) \cdot z, (x4 \cdot y3 + x5 \cdot y6 + x6 \cdot y9) \cdot z]]$

梯度计算时由于MindSpore采用的是Reverse[3]自动微分机制，会对输出结果求和后再对输入`x`求导：

(1) 求和公式：

$\sum{output} = [(x1 \cdot y1 + x2 \cdot y4 + x3 \cdot y7) + (x1 \cdot y2 + x2 \cdot y5 + x3 \cdot y8) + (x1 \cdot y3 + x2 \cdot y6 + x3 \cdot y9) +$

$(x4 \cdot y1 + x5 \cdot y4 + x6 \cdot y7) + (x4 \cdot y2 + x5 \cdot y5 + x6 \cdot y8) + (x4 \cdot y3 + x5 \cdot y6 + x6 \cdot y9)] \cdot z$

(2) 求导公式：

$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}x} = [[(y1 + y2 + y3) \cdot z，(y4 + y5 + y6) \cdot z，(y7 + y8 + y9) \cdot z]，[(y1 + y2 + y3) \cdot z，(y4 + y5 + y6) \cdot z，(y7 + y8 + y9) \cdot z]]$

(3) 计算结果：

$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}x} = [[4.5099998 \quad 2.7 \quad 3.6000001] [4.5099998 \quad 2.7 \quad 3.6000001]]$

若考虑对`x`、`y`输入求导，只需在`GradNetWrtX`中设置`self.grad_op = GradOperation(get_all=True)`。

### 权重求导

若考虑对权重的求导，将`GradNetWrtX`修改成：

```python
class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_by_list=True)
    def construct(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)
```

```python
output = GradNetWrtX(Net())(x, y)
print(output)
```

输出结果如下：

```text
(Tensor(shape=[1], dtype=Float32, value= [ 2.15359993e+01]),)
```

求导公式变为：

$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}z} = (x1 \cdot y1 + x2 \cdot y4 + x3 \cdot y7) + (x1 \cdot y2 + x2 \cdot y5 + x3 \cdot y8) + (x1 \cdot y3 + x2 \cdot y6 + x3 \cdot y9) + $

$(x4 \cdot y1 + x5 \cdot y4 + x6 \cdot y7) + (x4 \cdot y2 + x5 \cdot y5 + x6 \cdot y8) + (x4 \cdot y3 + x5 \cdot y6 + x6 \cdot y9)$

计算结果：

$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}z} = [2.15359993e+01]$

### 梯度值缩放

可以通过`sens_param`参数控制梯度值的缩放：

```python
class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(sens_param=True)
        self.grad_wrt_output = Tensor([[0.1, 0.6, 0.2], [0.8, 1.3, 1.1]], dtype=mstype.float32)
    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y, self.grad_wrt_output)
```

```python
output = GradNetWrtX(Net())(x, y)  
print(output)
```

输出结果如下：

```text
[[2.211 0.51 1.49 ]
 [5.588 2.68 4.07 ]]
```

`self.grad_wrt_output`可以记作如下形式：

```python
self.grad_wrt_output = Tensor([[s1, s2, s3], [s4, s5, s6]])
```

缩放后的输出值为原输出值与`self.grad_wrt_output`对应元素的乘积：

$output = [[(x1 \cdot y1 + x2 \cdot y4 + x3 \cdot y7) \cdot z \cdot s1，(x1 \cdot y2 + x2 \cdot y5 + x3 \cdot y8) \cdot z \cdot s2，(x1 \cdot y3 + x2 \cdot y6 + x3 \cdot y9) \cdot z \cdot s3]，$

$[(x4 \cdot y1 + x5 \cdot y4 + x6 \cdot y7) \cdot z \cdot s4，(x4 \cdot y2 + x5 \cdot y5 + x6 \cdot y8) \cdot z \cdot s5，(x4 \cdot y3 + x5 \cdot y6 + x6 \cdot y9) \cdot z \cdot s6]]$

求导公式变为输出值总和对`x`的每个元素求导：

$\frac{\mathrm{d}(\sum{output})}{\mathrm{d}x} = [[(s1 \cdot y1 + s2 \cdot y2 + s3 \cdot y3) \cdot z，(s1 \cdot y4 + s2 \cdot y5 + s3 \cdot y6) \cdot z，(s1 \cdot y7 + s2 \cdot y8 + s3 \cdot y9) \cdot z]，$

$[(s4 \cdot y1 + s5 \cdot y2 + s6 \cdot y3) \cdot z，(s4 \cdot y4 + s5 \cdot y5 + s6 \cdot y6) \cdot z，(s4 \cdot y7 + s5 \cdot y8 + s6 \cdot y9) \cdot z]]$

如果想计算单个输出（例如`output[0][0]`）对输入的导数，可以将相应位置的缩放值置为1，其他置为0；也可以改变网络结构：

```python
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')
    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out[0][0]
```

```python
output = GradNetWrtX(Net())(x, y)  
print(output)
```

输出结果如下：

```text
[[0.11 1.1 1.1]
 [0.   0.  0. ]]
```

## 停止计算梯度

我们可以使用`stop_gradient`来禁止网络内的算子对梯度的影响，例如：

```python
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import ParameterTuple, Parameter
from mindspore import dtype as mstype
from mindspore.ops import stop_gradient

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        out1 = self.matmul(x, y)
        out2 = self.matmul(x, y)
        out2 = stop_gradient(out2)
        out = out1 + out2
        return out

class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
output = GradNetWrtX(Net())(x, y)
print(output)
```

```text
    [[4.5, 2.7, 3.6],
     [4.5, 2.7, 3.6]]
```

在这里我们对`out2`设置了`stop_gradient`, 所以`out2`没有对梯度计算有任何的贡献。 如果我们删除`out2 = stop_gradient(out2)`，那么输出值会变为：

```text
    [[9.0, 5.4, 7.2],
     [9.0, 5.4, 7.2]]
```

在我们不对`out2`设置`stop_gradient`后， `out2`和`out1`会对梯度产生相同的贡献。 所以我们可以看到，结果中每一项的值都变为了原来的两倍。

## 高阶求导

高阶微分在AI支持科学计算、二阶优化等领域均有应用。如分子动力学模拟中，利用神经网络训练势能时[1]，损失函数中需计算神经网络输出对输入的导数，则反向传播便存在损失函数对输入、权重的二阶交叉导数；此外，AI求解微分方程（如PINNs[2]方法）还会存在输出对输入的二阶导数。又如二阶优化中，为了能够让神经网络快速收敛，牛顿法等需计算损失函数对权重的二阶导数。以下将主要介绍MindSpore图模式下的高阶导数。

MindSpore可通过多次求导的方式支持高阶导数，下面通过几类例子展开阐述。

完整样例代码见：[高阶求导样例代码](https://gitee.com/mindspore/docs/tree/r1.6/docs/sample_code/high_order_differentiation/second_order)

### 单输入单输出高阶导数

例如Sin算子，其二阶导数（-Sin）实现如下：

```python
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()
    def construct(self, x):
        out = self.sin(x)
        return out

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network
    def construct(self, x):
        gout= self.grad(self.network)(x)
        return gout
class GradSec(nn.Cell):
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation()
        self.network = network
    def construct(self, x):
        gout= self.grad(self.network)(x)
        return gout

net=Net()
firstgrad = Grad(net) # first order
secondgrad = GradSec(firstgrad) # second order
x_train = Tensor(np.array([1.0], dtype=np.float32))
output = secondgrad(x_train)
print(output)
```

输出结果如下：

```text
[-0.841471]
```

### 单输入多输出高阶导数

例如多输出的乘法运算，其高阶导数如下：

```python
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()
    def construct(self, x):
        out = self.mul(x, x)
        return out

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(sens_param=False)
        self.network = network
    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout
class GradSec(nn.Cell):
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation(sens_param=False)
        self.network = network
    def construct(self, x):
        gout = self.grad(self.network)(x)
        return gout

net=Net()
firstgrad = Grad(net) # first order
secondgrad = GradSec(firstgrad) # second order
x = Tensor([0.1, 0.2, 0.3], dtype=mstype.float32)
output = secondgrad(x)
print(output)
```

输出结果如下：

```text
[2. 2. 2.]
```

### 多输入多输出高阶导数

例如神经网络有多个输入`x`、`y`，可以通过梯度缩放机制获得二阶导数`dxdx`，`dydy`，`dxdy`，`dydx`如下：

```python
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()

    def construct(self, x, y):
        x_square = self.mul(x, x)
        x_square_y = self.mul(x_square, y)
        return x_square_y

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=False)
        self.network = network
    def construct(self, x, y):
        gout = self.grad(self.network)(x, y) # return dx, dy
        return gout

class GradSec(nn.Cell):
    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad = ops.GradOperation(get_all=True, sens_param=True)
        self.network = network
        self.sens1 = Tensor(np.array([1]).astype('float32'))
        self.sens2 = Tensor(np.array([0]).astype('float32'))
    def construct(self, x, y):
        dxdx, dxdy = self.grad(self.network)(x, y, (self.sens1,self.sens2))
        dydx, dydy = self.grad(self.network)(x, y, (self.sens2,self.sens1))
        return dxdx, dxdy, dydx, dydy

net = Net()
firstgrad = Grad(net) # first order
secondgrad = GradSec(firstgrad) # second order
x_train = Tensor(np.array([4],dtype=np.float32))
y_train = Tensor(np.array([5],dtype=np.float32))
dxdx, dxdy, dydx, dydy = secondgrad(x_train, y_train)
print(dxdx, dxdy, dydx, dydy)
```

输出结果如下：

```text
[10] [8.] [8.] [0.]
```

具体地，一阶导数计算的结果是`dx`、`dy`：如果计算`dxdx`，则一阶导数只需保留`dx`，对应`x`、`y`的缩放值分别设置成1和0，即`self.grad(self.network)(x, y, (self.sens1,self.sens2))`；同理计算`dydy`，则一阶导数只保留`dy`，对应`x`、`y`的`sens_param`分别设置成0和1，即`self.grad(self.network)(x, y, (self.sens2,self.sens1))`。

## 二阶微分算子支持情况

CPU支持算子：[Square](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Square.html#mindspore.ops.Square)、
[Exp](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Exp.html#mindspore.ops.Exp)、[Neg](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Neg.html#mindspore.ops.Neg)、[Mul](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Mul.html#mindspore.ops.Mul)、[MatMul](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.MatMul.html#mindspore.ops.MatMul)；

GPU支持算子：[Pow](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Pow.html#mindspore.ops.Pow)、[Log](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Log.html#mindspore.ops.Log)、[Square](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Square.html#mindspore.ops.Square)、[Exp](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Exp.html#mindspore.ops.Exp)、[Neg](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Neg.html#mindspore.ops.Neg)、[Mul](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Mul.html#mindspore.ops.Mul)、[Div](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Div.html#mindspore.ops.Div)、[MatMul](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.MatMul.html#mindspore.ops.MatMul)、[Sin](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Sin.html#mindspore.ops.Sin)、[Cos](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Cos.html#mindspore.ops.Cos)、[Tan](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Tan.html#mindspore.ops.Tan)、[Atanh](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Atanh.html#mindspore.ops.Atanh)；

Ascend支持算子：[Pow](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Pow.html#mindspore.ops.Pow)、[Log](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Log.html#mindspore.ops.Log)、[Square](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Square.html#mindspore.ops.Square)、[Exp](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Exp.html#mindspore.ops.Exp)、[Neg](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Neg.html#mindspore.ops.Neg)、[Mul](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Mul.html#mindspore.ops.Mul)、[Div](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Div.html#mindspore.ops.Div)、[MatMul](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.MatMul.html#mindspore.ops.MatMul)、[Sin](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Sin.html#mindspore.ops.Sin)、[Cos](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Cos.html#mindspore.ops.Cos)、[Tan](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Tan.html#mindspore.ops.Tan)、[Sinh](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Sinh.html#mindspore.ops.Sinh)、[Cosh](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Cosh.html#mindspore.ops.Cosh)、[Atanh](https://www.mindspore.cn/docs/api/zh-CN/r1.6/api_python/ops/mindspore.ops.Atanh.html#mindspore.ops.Atanh)。

## Jvp与Vjp接口

除了基于反向微分模式的GradOperation接口之外，MindSpore还提供了两个新的微分接口Vjp与Jvp，分别对应前向自动微分与反向自动微分。

### Jvp

Jvp(Jacobian-vector-product)对应的是前向模式的自动微分，适用在输入的维度小于输出的维度的网络中。Jvp会将输入网络的正向运行结果以及微分结果返回出来。不同于反向自动微分，前向自动微分可以在求取网络的原本输出的同时求取其梯度，不需要像反向微分一样保存太多的中间结果，因此前向自动微分相比于反向自动微分往往会节省一定的内存。反向微分与正向微分的区别可以详见[自动微分设计](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/design/gradient.html)。

样例代码如下：

```python
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()

    def construct(self, x, y):
        a = self.sin(x)
        b = self.cos(y)
        out = a + b
        return out

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.grad_op = nn.Jvp(net)

    def construct(self, x, y, v):
        output = self.grad_op(x, y, (v, v))
        return output

x = Tensor([0.8, 0.6, 0.2], dtype=mstype.float32)
y = Tensor([0.7, 0.4, 0.3], dtype=mstype.float32)
v = Tensor([1, 1, 1], dtype=mstype.float32)
output = GradNet(Net())(x, y, v)
print(output)
```

输出结果为：

```text
(Tensor(shape=[3], dtype=Float32, value= [ 1.48219836e+00, 1.48570347e+00, 1.15400589e+00]), Tensor(shape=[3], dtype=Float32, value= [ 5.24890423e-02,
4.35917288e-01, 6.84546351e-01]))
```

### Vjp

Vjp(Vector-jacobian-product), 运行的是反向模式的自动微分。Vjp会将输入网络的前向结果以及微分结果一并输出出来。 反向微分更加适用在输入的维度大于输出维度的网络中，具体内容详见[自动微分设计](https://www.mindspore.cn/docs/programming_guide/zh-CN/r1.6/design/gradient.html)。

样例代码如下：

```python
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import dtype as mstype
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()

    def construct(self, x, y):
        a = self.sin(x)
        b = self.cos(y)
        out = a + b
        return out

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.net = net
        self.grad_op = nn.Vjp(net)

    def construct(self, x, y, v):
        output = self.grad_op(x, y, v)
        return output

x = Tensor([0.8, 0.6, 0.2], dtype=mstype.float32)
y = Tensor([0.7, 0.4, 0.3], dtype=mstype.float32)
v = Tensor([1, 1, 1], dtype=mstype.float32)
output = GradNet(Net())(x, y, v)
print(output)
```

输出结果为：

```text
(Tensor(shape=[3], dtype=Float32, value= [ 1.48219836e+00, 1.48570347e+00, 1.15400589e+00]), (Tensor(shape=[3], dtype=Float32, value= [ 6.96706712e-01,
8.25335622e-01, 9.80066597e-01]), Tensor(shape=[3], dtype=Float32, value= [-6.44217670e-01, -3.89418334e-01, -2.95520216e-01])))
```

## 函数式接口grad、jvp和vjp

自动微分功能在科学计算领域有重要的使用场景，科学计算领域一般采用函数式的接口。为了提升自动微分功能的易用性，MindSpore提供GradOperation、Jvp和Vjp的函数式接口`grad`、`jvp`和`vjp`。函数式接口无需实例化被调用接口，契合用户的使用习惯。

### 函数式grad

`grad`接口用于生成输入函数的梯度。利用`grad_position`和`sens_param`参数控制梯度的计算方式。`grad_position`默认是`0`，表示只对第一个输入求导；为非零`int`类型时，会对该数字对应索引位置输入求导；为`tuple`时，会对tuple内数字索引位置的输入求导。`sens_param`默认是`False`，表示不对网络输出值缩放，当`sens_param`为`True`时，对网络输出值进行缩放。

样例代码如下：

`grad_position`参数控制对特定输入求导。

```python
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import grad
context.set_context(mode=context.GRAPH_MODE)
class Net(nn.Cell):
    def construct(self, x, y, z):
        return x*y*z

x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
net = Net()
output = grad(net, grad_position=(1, 2))(x, y, z)
print(output)
```

输出结果为：

```text
(Tensor(shape=[2, 2], dtype=Float32, value=
[[ 0.00000000e+00,  6.00000000e+00],
 [ 1.50000000e+01, -4.00000000e+00]]), Tensor(shape=[2, 2], dtype=Float32, value=
[[-2.00000000e+00,  6.00000000e+00],
 [-3.00000000e+00,  8.00000000e+00]]))
```

`sens_param`参数控制对网络输出进行缩放。

```python
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import grad
context.set_context(mode=context.GRAPH_MODE)
class Net(nn.Cell):
    def construct(self, x, y, z):
        return x**2 + y**2 + z**2, x*y*z

x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
y = Tensor(np.array([[-2, 3], [-1, 2]]).astype(np.float32))
z = Tensor(np.array([[0, 3], [5, -1]]).astype(np.float32))
v = Tensor(np.array([[-1, 3], [2, 1]]).astype(np.float32))
net = Net()
output = grad(net, grad_position=(1, 2), sens_param=True)(x, y, z, (v, v))
print(output)
```

输出结果为：

```text
(Tensor(shape=[2, 2], dtype=Float32, value=
[[ 4.00000000e+00,  3.60000000e+01],
 [ 2.60000000e+01,  0.00000000e+00]]), Tensor(shape=[2, 2], dtype=Float32, value=
[[ 2.00000000e+00,  3.60000000e+01],
 [ 1.40000000e+01,  6.00000000e+00]]))
```

### 函数式jvp

`jvp`对应前向模式的自动微分，输出网络的正向运算结果和正向微分结果。输出的元组的第一个元素为网络的正向运行结果，第二个元组为网络的正向微分结果。

样例代码如下：

```python
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops import jvp
from mindspore import Tensor
context.set_context(mode=context.GRAPH_MODE)
class Net(nn.Cell):
    def construct(self, x, y):
        return x**3 + y

x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
output = jvp(Net(), (x, y), (v, v))
print(output)
```

输出结果为：

```text
(Tensor(shape=[2, 2], dtype=Float32, value=
[[ 2.00000000e+00, 1.00000000e+01],
[ 3.00000000e+01, 6.80000000e+01]]), Tensor(shape=[2, 2], dtype=Float32, value=
[[ 4.00000000e+00, 1.30000000e+01],
[ 2.80000000e+01, 4.90000000e+01]]))
```

### 函数式vjp

`vjp`对应反向模式的自动微分，输出网络的正向运算结果和反向微分结果。输出的元组的第一个元素为网络的正向运行结果，第二个元组为网络的反向微分结果。

样例代码如下：

```python
import numpy as np
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops import vjp
from mindspore import Tensor
context.set_context(mode=context.GRAPH_MODE)
class Net(nn.Cell):
    def construct(self, x, y):
        return x**3 + y

x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
output = vjp(Net(), (x, y), v)
print(output)
```

输出结果为：

```text
(Tensor(shape=[2, 2], dtype=Float32, value=
[[ 2.00000000e+00, 1.00000000e+01],
[ 3.00000000e+01, 6.80000000e+01]]), (Tensor(shape=[2, 2], dtype=Float32, value=
[[ 3.00000000e+00, 1.20000000e+01],
[ 2.70000000e+01, 4.80000000e+01]]), Tensor(shape=[2, 2], dtype=Float32, value=
[[ 1.00000000e+00, 1.00000000e+00],
[ 1.00000000e+00, 1.00000000e+00]])))
```

## 引用

[1] Zhang L, Han J, Wang H, et al. [Deep potential molecular dynamics: a scalable model with the accuracy of quantum mechanics[J]](https://arxiv.org/pdf/1707.09571v2.pdf). Physical review letters, 2018, 120(14): 143001.

[2] Raissi M, Perdikaris P, Karniadakis G E. [Physics informed deep learning (part i): Data-driven solutions of nonlinear partial differential equations[J]](https://arxiv.org/pdf/1711.10561.pdf). arXiv preprint arXiv:1711.10561, 2017.

[3] Baydin A G, Pearlmutter B A, Radul A A, et al. [Automatic differentiation in machine learning: a survey[J]](https://jmlr.org/papers/volume18/17-468/17-468.pdf). The Journal of Machine Learning Research, 2017, 18(1): 5595-5637.
