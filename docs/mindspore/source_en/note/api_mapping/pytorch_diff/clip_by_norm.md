# Differences with torch.nn.utils.clip_grad_norm_

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/clip_by_norm.md)

## torch.nn.utils.clip_grad_norm_

```text
torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0)
```

For more information, see [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.clip_grad_norm_.html).

## mindspore.ops.clip_by_norm

```text
mindspore.ops.clip_by_norm(x, max_norm, norm_type=2.0, error_if_nonfinite=False)
```

For more information, see [mindspore.ops.clip_by_norm](https://www.mindspore.cn/docs/en/r2.3/api_python/ops/mindspore.ops.clip_by_norm.html).

## Differences

The gradient in PyTorch is a property of the Tensor and can be made to have a gradient by setting `requires_grad=True`. Due to the difference of framework mechanism, the gradient and weight are Tensor independent of each other in MindSpore. Therefore, when gradient cropping, MindSpore needs to obtain the gradient Tensor before cropping.

PyTorch: Gradient cropping can be implemented by directly passing in a Tensor with a gradient.

MindSpore: Due to the different framework mechanism, to implement gradient cropping, it is necessary to obtain the gradient first and then cropping the gradient. You can use methods such as `mindspore.grad` to obtain the gradient. For details, please refer to [gradient derivation](https://www.mindspore.cn/docs/en/r2.3/migration_guide/model_development/gradient.html#gradient-derivation).

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | parameters   | x        | The gradient mechanism is different. PyTorch can crop the gradient by passing in the Tensor, while MindSpore needs to pass in the gradient Tensor. Please refer to [gradient derivation](https://www.mindspore.cn/docs/en/r2.3/migration_guide/model_development/gradient.html#gradient-derivation) for how to obtain the gradient. |
|      | Parameter 2 | max_norm   | max_norm        |Consistent |
|      | Parameter 3 | norm_type| norm_type   | Consistent |
|      | Parameter 4 | -| error_if_nonfinite   |  PyTorch 1.12 version adds new parameters, with the same functionality as 1.12 |

### Code Example

> Due to the different mechanism, before implementing gradient cropping, MindSpore needs to use [mindspore.grad](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/mindspore.grad.html) and other methods to obtain the gradient first, (For more methods to obtain the gradient, please refer to [gradient derivation](https://www.mindspore.cn/docs/en/r2.3/migration_guide/model_development/gradient.html#gradient-derivation)), and then crop the gradient. The example code is as follows.

```python
import numpy as np
import torch
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer, Zero

data = np.array([0.2, 0.5, 0.2], dtype=np.float32)
label = np.array([1, 0], dtype=np.float32)
label_pt = np.array([0], dtype=np.float32)


# PyTorch
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
out = torch.unsqueeze(out, 0)
loss = loss_fun(out, torch.tensor(label_pt, dtype=torch.long))
loss.backward()
grads = [p.grad for p in net1.parameters() if p.grad is not None]
print(f"torch before grads:\n{grads}")
# torch before grads:
# [tensor([[-0.1000, -0.2500, -0.1000],
#         [ 0.1000,  0.2500,  0.1000]]), tensor([-0.5000,  0.5000])]
torch.nn.utils.clip_grad_norm_(net1.parameters(), max_norm=0.5)
print(f"torch after grads:\n{grads}")


# torch after grads:
# [tensor([[-0.0613, -0.1533, -0.0613],
#         [ 0.0613,  0.1533,  0.0613]]), tensor([-0.3066,  0.3066])]

# MindSpore
class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.dense = nn.Dense(3, 2)
        self.apply(self._init_weights)

    def _init_weights(self, cell):
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(initializer(Zero(), cell.weight.shape, cell.weight.dtype))
            cell.bias.set_data(initializer(Zero(), cell.bias.shape, cell.bias.dtype))

    def construct(self, x):
        return self.dense(x)


net2 = Net2()
loss_fn = nn.CrossEntropyLoss()


def forward_fn(data, label):
    logits = ms.ops.squeeze(net2(data))
    loss = loss_fn(logits, label)
    return loss, logits


grad_fn = ms.grad(forward_fn, grad_position=None, weights=net2.trainable_params(), has_aux=True)
grads, out2 = grad_fn(ms.ops.unsqueeze(ms.Tensor(data), dim=0), ms.Tensor(label))
print(f"ms before grads:\n{grads}")
# ms before grads:
# (Tensor(shape=[2, 3], dtype=Float32, value=
# [[-1.00000001e-01, -2.50000000e-01, -1.00000001e-01],
#  [ 1.00000001e-01,  2.50000000e-01,  1.00000001e-01]]), Tensor(shape=[2], dtype=Float32, value= [-5.00000000e-01,  5.00000000e-01]))
grads = ms.ops.clip_by_norm(grads, max_norm=0.5)
print(f'ms after grads:\n{grads}')
# ms after grads:
# (Tensor(shape=[2, 3], dtype=Float32, value=
# [[-6.13138638e-02, -1.53284654e-01, -6.13138638e-02],
#  [ 6.13138638e-02,  1.53284654e-01,  6.13138638e-02]]), Tensor(shape=[2], dtype=Float32, value= [-3.06569308e-01,  3.06569308e-01]))
```
