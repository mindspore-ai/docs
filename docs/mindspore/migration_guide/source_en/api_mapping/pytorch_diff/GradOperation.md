# Function Differences with torch.autograd.backward and torch.autograd.grad

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/GradOperation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

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

## mindspore.ops.GradOperation

```python
class mindspore.ops.GradOperation(
  get_all=False,
  get_by_list=False,
  sens_param=False
)
```

For more information, see [mindspore.ops.GradOperation](https://mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.GradOperation.html#mindspore.ops.GradOperation).

## Differences

PyTorch: Use `torch.autograd.backward` to compute the sum of gradients of given Tensors with respect to graph leaves. When calculating the gradient of the Tensor with backpropagation, only the gradient of graph leaves with `requires_grad=True` will be calculated. Use `torch.autograd.grad` to compute and return the sum of gradients of outputs with respect to the inputs. If `only_inputs` is True, the function will only return a list of gradients with respect to the specified inputs.

MindSpore: Compute the first derivative. When `get_all` is set to False, the first input derivative is computed. When `get_all` is set to True, all input derivatives are computed. When `get_by_list` is set to False, weight derivatives are not computed. When `get_by_list` is set to True, the weight derivative is computed. `sens_param` scales the output value of the network to change the final gradient.

## Code Example

```python
# In MindSpore：
import numpy as np
import mindspore.nn as nn
from mindspore import dtype as mstype
from mindspore import ops, Tensor, Parameter

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

x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
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
