# Function Differences with torch.autograd.backward and torch.autograd.grad

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/pytorch_diff/grad.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [torch.autograd.backward](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.backward).

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

For more information, see [torch.autograd.grad](https://pytorch.org/docs/1.5.0/autograd.html#torch.autograd.grad).

## mindspore.grad

```python
mindspore.grad(
  fn,
  grad_position=0,
  weights=None,
  has_aux=False
)
```

For more information, see [mindspore.grad](https://www.mindspore.cn/docs/en/r2.0.0-alpha/api_python/mindspore/mindspore.grad.html).

## Differences

PyTorch: Use `torch.autograd.backward` to compute the sum of gradients of given Tensors with respect to graph leaves. When calculating the gradient of the Tensor with backpropagation, only the gradient of graph leaves with `requires_grad=True` will be calculated. Use `torch.autograd.grad` to compute and return the sum of gradients of outputs with respect to the inputs. If `only_inputs` is True, the function will only return a list of gradients with respect to the specified inputs.

MindSpore: Compute the first derivative. When `grad_position` is set to int or tuple of int, the corresponding input derivatives are computed. if `weights` is set, the network parameters derivatives will be computed. If `has_aux` is True,  only the first output of `fn` participates in the computation, in this case, the `fn` should has at least two outputs.

## Code Example

```python
# In MindSpore：
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops

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
