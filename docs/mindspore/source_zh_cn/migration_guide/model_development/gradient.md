# 梯度求导

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/migration_guide/model_development/gradient.md)

## 自动微分对比

MindSpore 和 PyTorch 都提供了自动微分功能，让我们在定义了正向网络后，可以通过简单的接口调用实现自动反向传播以及梯度更新。但需要注意的是，MindSpore 和 PyTorch 构建反向图的逻辑是不同的，这个差异也会带来 API 设计上的不同。

<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch的自动微分 </td> <td style="text-align:center"> MindSpore的自动微分 </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
# torch.autograd:
# backward是累计的，更新完之后需清空optimizer

import torch
from torch.autograd import Variable

x = Variable(torch.ones(2, 2),
             requires_grad=True)
x = x * 2
y = x - 1
y.backward(x)

```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
# ms.grad:
# 使用grad接口，输入正向图，输出反向图
import mindspore as ms
from mindspore import nn
class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net

    def construct(self, x, y):
        gradient_function = ms.grad(self.net)
        return gradient_function(x, y)
```

</pre>
</td>
</tr>
</table>

### 原理对比

#### PyTorch的自动微分

我们知道 PyTorch 是基于计算路径追踪的自动微分，当我们定义一个网络结构后， 并不会建立反向图，而是在执行正向图的过程中，`Variable` 或 `Parameter` 记录每一个正向计算对应的反向函数，并生成一个动态计算图，用于后续的梯度计算。当在最终的输出处调用 `backward` 时，就会从根节点到叶节点应用链式法则计算梯度。PyTorch 的动态计算图所存储的节点实际是 `Function` 函数对象，每当对 `Tensor` 执行一步运算后，就会产生一个 `Function` 对象，它记录了反向传播中必要的信息。反向传播过程中，`autograd` 引擎会按照逆序，通过 `Function` 的 `backward` 依次计算梯度。 这一点我们可以通过 `Tensor` 的隐藏属性查看。

#### MindSpore的自动微分

在图模式下，MindSpore 的自动微分是基于图结构的微分，和 PyTorch 不同，它不会在正向计算过程中记录任何信息，仅仅执行正常的计算流程（在PyNative模式下和 PyTorch 类似）。那么问题来了，如果整个正向计算都结束了，MindSpore 也没有记录任何信息，那它是如何知道反向传播怎么执行的呢？

MindSpore 在做自动微分时，通过对正向图的分析得到反向传播信息，其结果与正向计算中具体的数值无关，仅和正向图结构有关。通过对正向图的自动微分，我们得到了反向图。将反向图添加到用户定义的正向图之后，组成一个最终的计算图。不过后添加的反向图和其中的反向算子我们并不感知，也无法手动添加，只能通过 MindSpore 为我们提供的接口自动添加，这样做也避免了我们在反向构图时引入错误。

最终，我们看似仅执行了正向图，其实图结构里既包含了正向算子，又包含了 MindSpore 为我们添加的反向算子，也就是说，MindSpore 在我们定义的正向图后面又新加了一个看不见的  `Cell`，这个  `Cell` 里都是根据正向图推导出来的反向算子。

而这个帮助我们构建反向图的接口就是 [grad](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.grad.html) 。

通过`grad`接口得到反向图之后，对于输入的任何一组数据，不仅能计算正向输出，还能计算所有权重的梯度。由于图结构固定，不保存中间变量，所以这个新计算图可以被反复调用。

同理，之后我们再给网络加上优化器结构时，优化器也会加上优化器相关的算子，也就是再给这个计算图加上我们不感知的优化器算子，最终，计算图就构建完成。

在 MindSpore 中，大部分操作都会最终转换成真实的算子操作，最终加入到计算图中，因此，我们实际执行的计算图中算子的数量远多于我们最初定义的计算图中算子的数量。

在MindSpore中，提供了[TrainOneStepCell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.TrainOneStepCell.html)和[TrainOneStepWithLossScaleCell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.TrainOneStepWithLossScaleCell.html)这两个接口来包装整个训练流程，如果在常规的训练流程外有其他的操作，如梯度裁剪、规约、中间变量返回等，需要自定义训练的Cell，详情请参考[训练及推理流程](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/training_and_evaluation.html)。

### 接口对比

#### torch.autograd.backward

[torch.autograd.backward](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html)对于一个标量，调用它的backward方法后会根据链式法则自动计算出叶子节点的梯度值。对于向量和矩阵，需要定义grad_tensor来计算矩阵的梯度。
通常在调用一次backward后，PyTorch会自动把计算图销毁，所以要想对某个变量重复调用backward，则需要将retain_graph参数设置为True。
若需要计算更高阶的梯度，需要将create_graph设置为True。
z.backward()和torch.autograd.backward(z)两种表达等价。

该接口在MindSpore中用mindspore.grad实现。上述PyTorch用例可转化为：

<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
# 在调用backward函数之前，x.grad和y.grad函数为空
# backward计算过后，x.grad和y.grad分别代表导数计算后的值
import torch
print("=== tensor.backward ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
print("x.grad before backward", x.grad)
print("y.grad before backward", y.grad)
z.backward()
print("z", z)
print("x.grad", x.grad)
print("y.grad", y.grad)
print("=== torch.autograd.backward ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
torch.autograd.backward(z)
print("z", z)
print("x.grad", x.grad)
print("y.grad", y.grad)
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
import mindspore
print("=== mindspore.grad ===")
x = mindspore.Tensor(1.0)
y = mindspore.Tensor(2.0)
def net(x, y):
    return x**2+y
out = mindspore.grad(net, grad_position=0)(x, y)
print("out", out)
out1 = mindspore.grad(net, grad_position=1)(x, y)
print("out1", out1)
```

</pre>
</td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

运行结果：

```text
=== tensor.backward ===
x.grad before backward None
y.grad before backward None
z tensor(3., grad_fn=<AddBackward0>)
x.grad tensor(2.)
y.grad tensor(1.)
=== torch.autograd.backward ===
z tensor(3., grad_fn=<AddBackward0>)
x.grad tensor(2.)
y.grad tensor(1.)
```

</pre>
</td>
<td style="vertical-align:top"><pre>

运行结果：

```text
=== mindspore.grad ===
out 2.0
out1 1.0
```

</pre>
</td>
</tr>
</table>

若上述net有多个输出，需要注意网络多输出对于求梯度的影响。

<table class="colwidths-auto docutils align-default">
<tr>
<td style="text-align:center"> PyTorch </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

```python
# 不支持多个输出
import torch
print("=== torch.autograd.backward 不支持多个output ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
torch.autograd.backward(z)

print("z", z)
print("x.grad", x.grad)
print("y.grad", y.grad)
```

</pre>
</td>
<td style="vertical-align:top"><pre>

```python
# 支持多个输出
import mindspore
print("=== mindspore.grad 多个output ===")
x = mindspore.Tensor(1.0)
y = mindspore.Tensor(2.0)
def net(x, y):
    return x**2+y, x
out = mindspore.grad(net, grad_position=0)(x, y)
print("out", out)
out1 = mindspore.grad(net, grad_position=1)(x, y)
print("out1", out)
```

</pre>
</td>
</tr>
<tr>
<td style="vertical-align:top"><pre>

运行结果：

```text
=== torch.autograd.backward 不支持多个output ===
z tensor(3., grad_fn=<AddBackward0>)
x.grad tensor(2.)
y.grad tensor(1.)
```

</pre>
</td>
<td style="vertical-align:top"><pre>

运行结果：

```text
=== mindspore.grad 多个output ===
out 3.0
out1 3.0
```

</pre>
</td>
</tr>
</table>

因此， 若要在MindSpore只对第一个输出求梯度，在MindSpore中需要使用has_aux参数。

```python
import mindspore
print("=== mindspore.grad has_aux ===")
x = mindspore.Tensor(1.0)
y = mindspore.Tensor(2.0)
def net(x, y):
    return x**2+y, x
grad_fcn = mindspore.grad(net, grad_position=0, has_aux=True)
out, _ = grad_fcn(x, y)
print("out", out)
grad_fcn1 = mindspore.grad(net, grad_position=1, has_aux=True)
out, _ = grad_fcn1(x, y)
print("out", out)
```

运行结果：

```text
=== mindspore.grad has_aux ===
out 2.0
out 1.0
```

#### torch.autograd.grad

[torch.autograd.grad](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html)此接口与torch.autograd.backward基本一致。两者的区别为：前者是直接修改各个 Tensor 的 grad 属性，后者是返回参数的梯度值列表。因此在迁移到MindSpore时，可同样参考上述用例。

```python
import torch
print("=== torch.autograd.grad ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
out = torch.autograd.grad(z, x)
out1 = torch.autograd.grad(z, y)
print("out", out)
print("out1", out1)
```

运行结果：

```text
=== torch.autograd.grad ===
out (tensor(2.),)
out1 (tensor(1.),)
```

#### torch.no_grad

在 PyTorch 中，默认情况下，执行正向计算时会记录反向传播所需的信息，在推理阶段或无需反向传播网络中，这一操作是冗余的，会额外耗时，因此，PyTorch 提供了`torch.no_grad` 来取消该过程。

而 MindSpore 只有在调用`grad`才会根据正向图结构来构建反向图，正向执行时不会记录任何信息，所以 MindSpore 并不需要该接口，也可以理解为 MindSpore 的正向计算均在`torch.no_grad` 情况下进行的。

```python
import torch
print("=== torch.no_grad ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x**2+y
print("z.requires_grad", z.requires_grad)
with torch.no_grad():
    z = x**2+y
print("z.requires_grad", z.requires_grad)
```

运行结果：

```text
=== torch.no_grad ===
z.requires_grad True
z.requires_grad False
```

#### torch.enable_grad

若 PyTorch 开启了 `torch.no_grad` 禁用了梯度计算，可以使用此接口启用。

而 MindSpore 只有在调用`grad`才会根据正向图结构来构建反向图，正向执行时不会记录任何信息，所以 MindSpore 并不需要该接口，也可以理解为 MindSpore 的反向计算均在`torch.enable_grad` 情况下进行的。

```python
import torch
print("=== torch.enable_grad ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    z = x**2+y
print("z.requires_grad", z.requires_grad)
with torch.enable_grad():
    z = x**2+y
print("z.requires_grad", z.requires_grad)
```

运行结果：

```text
=== torch.enable_grad ===
z.requires_grad False
z.requires_grad True
```

#### retain_graph

由于 PyTorch 是基于函数式的自动微分，所以默认每次执行完反向传播后都会自动清除记录的信息，从而进行下一次迭代。这就会导致当我们想再次利用这些反向图和梯度信息时，由于已被删除而获取失败。因此，PyTorch 提供了`backward(retain_graph=True)` 来主动保留这些信息。

而 MindSpore 则不需要这个功能，MindSpore 是基于计算图的自动微分，反向图信息在调用`grad`后便永久的记录在计算图中，只要再次调用计算图就可以获取梯度信息。

## MindSpore自动微分接口

本节介绍MindSpore提供的三种[自动微分](https://mindspore.cn/tutorials/zh-CN/master/beginner/autograd.html)接口用以计算模型的梯度结果。
在[自动求导](https://mindspore.cn/tutorials/zh-CN/master/advanced/derivation.html)的教程中，对各种梯度计算的场景做了一些介绍。

### mindspore.grad

[mindspore.grad](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.grad.html)这个API有四个可以配置的参数：

- fn (Union[Cell, Function]) - 待求导的函数或网络（Cell）。

- grad_position (Union[NoneType, int, tuple[int]]) - 指定求导输入位置的索引，默认值：0。

- weights (Union[ParameterTuple, Parameter, list[Parameter]]) - 训练网络中需要返回梯度的网络参数，默认值：None。

- has_aux (bool) - 是否返回辅助参数的标志。若为True， fn 输出数量必须超过一个，其中只有 fn 第一个输出参与求导，其他输出值将直接返回。默认值：False。

其中`grad_position`和`weights`共同决定要输出哪些值的梯度，has_aux在有多个输出时配置对第一个输入求梯度还是全部输出求梯度。

| grad_position | weights | output |
| ------------- | ------- | ------ |
| 0         | None   | 第一个输入的梯度 |
| 1         | None   | 第二个输入的梯度 |
| (0, 1)      | None   | (第一个输入的梯度, 第二个输入的梯度) |
| None       | weights | (weights的梯度) |
| 0         | weights | (第一个输入的梯度), (weights的梯度) |
| (0, 1)      | weights | (第一个输入的梯度, 第二个输入的梯度), (weights的梯度) |
| None       | None   | 报错  |

下面实际运行一个示例，看下具体是怎么用的。

首先，构造一个带参数的网络，这个网络有两个输出loss和logits，其中loss是我们用于求梯度的输出。

```python
import mindspore as ms
from mindspore import nn

class Net(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(Net, self).__init__()
        self.fc = nn.Dense(in_channel, out_channel, has_bias=False)
        self.loss = nn.MSELoss()

    def construct(self, x, y):
        logits = self.fc(x).squeeze()
        loss = self.loss(logits, y)
        return loss, logits

net = Net(3, 1)
net.fc.weight.set_data(ms.Tensor([[2, 3, 4]], ms.float32))   # 给全连接的weight设置固定值

print("=== weight ===")
for param in net.trainable_params():
    print("name:", param.name, "data:", param.data.asnumpy())
x = ms.Tensor([[1, 2, 3]], ms.float32)
y = ms.Tensor(19, ms.float32)

loss, logits = net(x, y)
print("=== output ===")
print(loss, logits)
```

运行结果：

```text
=== weight ===
name: fc.weight data: [[2. 3. 4.]]
=== output ===
1.0 20.0
```

```python
# 对第一个输入求梯度

print("=== grads 1 ===")
grad_func = ms.grad(net, grad_position=0, weights=None, has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)
```

运行结果：

```text
=== grads 1 ===
grad [[4. 6. 8.]]
logit (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# 对第二个输入求梯度

print("=== grads 2 ===")
grad_func = ms.grad(net, grad_position=1, weights=None, has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)
```

运行结果：

```text
=== grads 2 ===
grad -2.0
logit (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# 对多个输入求梯度

print("=== grads 3 ===")
grad_func = ms.grad(net, grad_position=(0, 1), weights=None, has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)
```

运行结果：

```text
=== grads 3 ===
grad (Tensor(shape=[1, 3], dtype=Float32, value=
[[4.00000000e+000, 6.00000000e+000, 8.00000000e+000]]), Tensor(shape=[], dtype=Float32, value= -2))
logit (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# 对weights求梯度

print("=== grads 4 ===")
grad_func = ms.grad(net, grad_position=None, weights=net.trainable_params(), has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logits", logit)
```

运行结果：

```text
=== grads 4 ===
grad (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),)
logits (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# 对第一个输入和weights求梯度

print("=== grads 5 ===")
grad_func = ms.grad(net, grad_position=0, weights=net.trainable_params(), has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)
```

运行结果：

```text
=== grads 5 ===
grad (Tensor(shape=[1, 3], dtype=Float32, value=
[[4.00000000e+000, 6.00000000e+000, 8.00000000e+000]]), (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),))
logit (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# 对多个输入和weights求梯度

print("=== grads 6 ===")
grad_func = ms.grad(net, grad_position=(0, 1), weights=net.trainable_params(), has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)
```

运行结果：

```text
=== grads 6 ===
grad ((Tensor(shape=[1, 3], dtype=Float32, value=
[[4.00000000e+000, 6.00000000e+000, 8.00000000e+000]]), Tensor(shape=[], dtype=Float32, value= -2)), (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),))
logit (Tensor(shape=[], dtype=Float32, value= 20),)
```

```python
# has_aux=False的场景

print("=== grads 7 ===")
grad_func = ms.grad(net, grad_position=0, weights=None, has_aux=False)
grad = grad_func(x, y)  # 只有一个输出
print("grad", grad)
```

运行结果：

```text
=== grads 7 ===
grad [[ 6.  9. 12.]]
```

`has_aux=False`的场景实际上等价于两个输出相加作为求梯度的输出：

```python
class Net2(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.fc = nn.Dense(in_channel, out_channel, has_bias=False)
        self.loss = nn.MSELoss()

    def construct(self, x, y):
        logits = self.fc(x).squeeze()
        loss = self.loss(logits, y)
        return loss + logits

net2 = Net2(3, 1)
net2.fc.weight.set_data(ms.Tensor([[2, 3, 4]], ms.float32))   # 给全连接的weight设置固定值
grads = ms.grad(net2, grad_position=0, weights=None, has_aux=False)
grad = grads(x, y)  # 只有一个输出
print("grad", grad)
```

运行结果：

```text
grad [[ 6.  9. 12.]]
```

```python
# grad_position=None, weights=None

print("=== grads 8 ===")
grad_func = ms.grad(net, grad_position=None, weights=None, has_aux=True)
grad, logit = grad_func(x, y)
print("grad", grad)
print("logit", logit)

# === grads 8 ===
# ValueError: `grad_position` and `weight` can not be None at the same time.
```

### mindspore.value_and_grad

[mindspore.value_and_grad](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.value_and_grad.html)这个接口和上面的grad的参数是一样的，只不过这个接口可以一次性计算网络的正向结果和梯度。

| grad_position | weights | output |
| ------------- | ------- | ------ |
| 0         | None   | (网络的输出, 第一个输入的梯度) |
| 1         | None   | (网络的输出, 第二个输入的梯度) |
| (0, 1)      | None   | (网络的输出, (第一个输入的梯度, 第二个输入的梯度)) |
| None       | weights | (网络的输出, (weights的梯度)) |
| 0         | weights | (网络的输出, ((第一个输入的梯度), (weights的梯度))) |
| (0, 1)      | weights | (网络的输出, ((第一个输入的梯度, 第二个输入的梯度), (weights的梯度))) |
| None       | None   | 报错  |

```python
print("=== value and grad ===")
value_and_grad_func = ms.value_and_grad(net, grad_position=(0, 1), weights=net.trainable_params(), has_aux=True)
value, grad = value_and_grad_func(x, y)
print("value", value)
print("grad", grad)
```

运行结果：

```text
=== value and grad ===
value (Tensor(shape=[], dtype=Float32, value= 1), Tensor(shape=[], dtype=Float32, value= 20))
grad ((Tensor(shape=[1, 3], dtype=Float32, value=
[[4.00000000e+000, 6.00000000e+000, 8.00000000e+000]]), Tensor(shape=[], dtype=Float32, value= -2)), (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),))
```

### mindspore.ops.GradOperation

[mindspore.ops.GradOperation](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.GradOperation.html)一个高阶函数，为输入函数生成梯度函数。

由 GradOperation 高阶函数生成的梯度函数可以通过构造参数自定义。

这个函数和grad的功能差不多，当前版本不推荐使用，详情请参考API内描述。

## loss scale

由于在混合精度的场景，在求梯度的过程中可能会遇到梯度下溢，一般我们会使用loss scale配套梯度求导使用。

> 在Ascend上因为Conv、Sort、TopK等算子只能是float16的，MatMul由于性能问题最好也是float16的，所以建议loss scale操作作为网络训练的标配。[Ascend 上只支持float16的算子列表](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/debug_and_tune.html#4%E8%AE%AD%E7%BB%83%E7%B2%BE%E5%BA%A6)。
>
> 溢出可以通过MindSpore Insight的[调试器](https://www.mindspore.cn/mindinsight/docs/zh-CN/master/debugger.html)或者[dump数据](https://mindspore.cn/tutorials/experts/zh-CN/master/debug/dump.html)获取到溢出算子信息。
>
> 一般溢出表现为loss Nan/INF，loss突然变得很大等。

```python
from mindspore.amp import StaticLossScaler, all_finite

loss_scale = StaticLossScaler(1024.)  # 静态lossscale

def forward_fn(x, y):
    loss, logits = net(x, y)
    print("loss", loss)
    loss = loss_scale.scale(loss)
    return loss, logits

value_and_grad_func = ms.value_and_grad(forward_fn, grad_position=None, weights=net.trainable_params(), has_aux=True)
(loss, logits), grad = value_and_grad_func(x, y)
print("=== loss scale ===")
print("loss", loss)
print("grad", grad)
print("=== unscale ===")
loss = loss_scale.unscale(loss)
grad = loss_scale.unscale(grad)
print("loss", loss)
print("grad", grad)

# 检查是否溢出，无溢出的话返回True
state = all_finite(grad)
print(state)
```

运行结果：

```text
loss 1.0
=== loss scale ===
loss 1024.0
grad (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.04800000e+003, 4.09600000e+003, 6.14400000e+003]]),)
=== unscale ===
loss 1.0
grad (Tensor(shape=[1, 3], dtype=Float32, value=
[[2.00000000e+000, 4.00000000e+000, 6.00000000e+000]]),)
True
```

loss scale的原理非常简单，通过给loss乘一个比较大的值，通过梯度的链式传导，在计算梯度的链路上乘一个比较大的值，防止在梯度反向传播过程中过小而出现精度问题。

在计算完梯度之后，需要把loss和梯度除回原来的值，保证整个计算过程正确。

最后一般需要使用all_finite来判断下是否有溢出，如果没有溢出的话就可以使用优化器进行参数更新了。

## 梯度裁剪

当训练过程中遇到梯度爆炸或者梯度特别大，训练不稳定的情况，可以考虑添加梯度裁剪，这里对常用的使用global_norm进行梯度裁剪的场景举例说明：

```python
from mindspore import ops

grad = ops.clip_by_global_norm(grad)
```

## 梯度累加

梯度累加是一种训练神经网络的数据样本按Batch拆分为几个小Batch的方式，然后按顺序计算，用以解决由于内存不足，导致Batch size过大，神经网络无法训练或者网络模型过大无法加载的OOM（Out Of Memory）问题。

详情请参考[梯度累加](https://www.mindspore.cn/tutorials/experts/zh-CN/master/optimize/gradient_accumulation.html)。
