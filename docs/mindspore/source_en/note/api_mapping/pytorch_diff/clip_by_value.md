# Function Differences with torch.nn.utils.clip_grad_value_

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/clip_by_value.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.utils.clip_grad_value_

```text
torch.nn.utils.clip_grad_value_(parameters, clip_value)
```

For more information, see [torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.clip_grad_value_.html).

## mindspore.ops.clip_by_value

```text
mindspore.ops.clip_by_value(x, clip_value_min=None, clip_value_max=None)
```

For more information, see [mindspore.ops.clip_by_value](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.clip_by_value.html).

## Differences

The gradient in PyTorch is a property of the Tensor and can be made to have a gradient by setting `requires_grad=True`. Due to the difference of framework mechanism, the gradient and weight are Tensor independent of each other in MindSpore. Therefore, when gradient cropping, MindSpore needs to obtain the gradient Tensor before cropping.

PyTorch: Gradient cropping can be implemented by directly passing in a Tensor with a gradient.

MindSpore: Due to the different framework mechanism, to implement gradient cropping, it is necessary to obtain the gradient first and then cropping the gradient. You can use methods such as `mindspore.grad` to obtain the gradient. For details, please refer to [gradient derivation](https://www.mindspore.cn/docs/en/master/migration_guide/model_development/gradient.html#gradient-derivation).

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | parameters   | x        | The gradient mechanism is different. PyTorch can crop the gradient by passing in the Tensor, while MindSpore needs to pass in the gradient Tensor. Please refer to [gradient derivation](https://www.mindspore.cn/docs/en/master/migration_guide/model_development/gradient.html#gradient-derivation) for how to obtain the gradient. |
|      | Parameter 2 | clip_value   | clip_value_min        | PyTorch cropping range is [-clip_value, clip_value], while MindSpore cropping range is [clip_value_min, clip_value_max] |
|      | Parameter 3 | -            | clip_value_max   |  PyTorch cropping range is [-clip_value, clip_value], while MindSpore cropping range is [clip_value_min, clip_value_max] |

### Code Example

> Due to the different mechanism, MindSpore needs to implement gradient cropping first using [mindspore.grad](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.grad.html) and other method to obtain the gradient, (For more methods to obtain the gradient, please refer to [gradient derivation](https://www.mindspore.cn/docs/en/master/migration_guide/model_development/gradient.html#gradient-derivation)), and then crop the gradient. The example code is as follows.

```python
import numpy as np

data = np.array([0.2, 0.5, 0.2], dtype=np.float32)
label = np.array([1, 0], dtype=np.float32)
# PyTorch
import torch
class Net1(torch.nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.dense = torch.nn.Linear(3, 2)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.zero_()
            module.bias.data.zero_()

    def forward(self, x):
        x = self.dense(x)
        return x

net1 = Net1()
loss_fun = torch.nn.CrossEntropyLoss()
out = net1(torch.tensor(data))
loss = loss_fun(out, torch.tensor(label))
loss.backward()
grads = [p.grad for p in net1.parameters() if p.grad is not None]
print(grads)
# Before clip out:
# [tensor([[-0.1000, -0.2500, -0.1000],
#         [ 0.1000,  0.2500,  0.1000]]), tensor([-0.5000,  0.5000])]
torch.nn.utils.clip_grad_value_(net1.parameters(), clip_value=0.1)
print(grads)
# After clip out:
# [tensor([[-0.1000, -0.1000, -0.1000],
#         [ 0.1000,  0.1000,  0.1000]]), tensor([-0.1000,  0.1000])]

# MindSpore
import mindspore as ms
from mindspore.common.initializer import initializer, Zero
class Net2(ms.nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.dense = ms.nn.Dense(3, 2)
        self.apply(self._init_weights)

    def _init_weights(self, cell):
        if isinstance(cell, ms.nn.Dense):
            cell.weight.set_data(initializer(Zero(), cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        return self.dense(x)

net2 = Net2()
loss_fn = ms.nn.CrossEntropyLoss()

def forward_fn(data, label):
    logits = net2(data)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = ms.grad(forward_fn, grad_position=None, weights=net2.trainable_params(), has_aux=True)
grads = grad_fn(ms.Tensor(data), ms.Tensor(label))
print(grads)
# Before clip out:
# ((Tensor(shape=[2, 3], dtype=Float32, value=
# [[-1.00000001e-01, -2.50000000e-01, -1.00000001e-01],
#  [ 1.00000001e-01,  2.50000000e-01,  1.00000001e-01]]), Tensor(shape=[2], dtype=Float32, value= [-5.00000000e-01,  5.00000000e-01])), (Tensor(shape=[2], dtype=Float32, value= [ 0.00000000e+00,  0.00000000e+00]),))
grads = ms.ops.clip_by_value(grads, clip_value_min=-0.1, clip_value_max=0.1)
print(grads)
# After clip out:
# ((Tensor(shape=[2, 3], dtype=Float32, value=
# [[-1.00000001e-01, -1.00000001e-01, -1.00000001e-01],
#  [ 1.00000001e-01,  1.00000001e-01,  1.00000001e-01]]), Tensor(shape=[2], dtype=Float32, value= [-1.00000001e-01,  1.00000001e-01])), (Tensor(shape=[2], dtype=Float32, value= [ 0.00000000e+00,  0.00000000e+00]),))
```
