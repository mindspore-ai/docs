# 比较与torch.autograd.backward和torch.autograd.grad的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/grad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.autograd.backward

```python
torch.autograd.backward(
  tensors,
  grad_tensors=None,
  retain_graph=None,
  create_graph=False,
  grad_variables=None
)
```

更多内容详见[torch.autograd.backward](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.backward)。

## torch.autograd.grad

```python
torch.autograd.grad(
  outputs,
  inputs,
  grad_outputs=None,
  retain_graph=None,
  create_graph=False,
  only_inputs=True,
  allow_unused=False
)
```

更多内容详见[torch.autograd.grad](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.grad)。

## mindspore.grad

```python
mindspore.grad(
  fn,
  grad_position=0,
  weights=None,
  has_aux=False
)
```

更多内容详见[mindspore.grad](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore/mindspore.grad.html)。

## 使用方式

PyTorch：使用`torch.autograd.backward`计算给定Tensor关于叶子节点的梯度总和，反向传播计算Tensor的梯度时，只计算`requires_grad=True`的叶子节点的梯度。使用`torch.autograd.grad`计算并返回输出关于输入的梯度总和，如果`only_inputs`为True，仅返回与指定输入相关的梯度列表。

MindSpore：计算梯度，当`grad_position`设置为int或者tuple int类型，将会计算对应输入位置的梯度。如果设置了`weights`, 将会计算网络的变量的参数。当`has_aux`设置为True时， 只有`fn`的第一个输出参与梯度计算， 此时`fn`至少具备两个输出。

## 代码示例

```python
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import ops

# In MindSpore：
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.matmul = ops.MatMul()
        self.z = ms.Parameter(ms.Tensor(np.array([1.0], np.float32)), name='z')
    def construct(self, x, y):
        x = x * self.z
        out = self.matmul(x, y)
        return out

class GradNetWrtX(nn.Cell):
    def __init__(self, net):
        super(GradNetWrtX, self).__init__()
        self.net = net
    def construct(self, x, y):
        gradient_function = ms.grad(self.net)
        return gradient_function(x, y)

x = ms.Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=ms.float32)
y = ms.Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=ms.float32)
output = GradNetWrtX(Net())(x, y)
print(output)
# Out:
# [[1.4100001 1.5999999 6.6      ]
#  [1.4100001 1.5999999 6.6      ]]

# In torch:
import torch
x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
z = x * x * y
z.backward()
print(x.grad, y.grad)
# Out:
# tensor(12.) tensor(4.)

x = torch.tensor(2.).requires_grad_()
y = torch.tensor(3.).requires_grad_()
z = x * x * y
grad_x = torch.autograd.grad(outputs=z, inputs=x)
print(grad_x[0])
# Out:
# tensor(12.)
```
