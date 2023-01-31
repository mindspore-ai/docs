# 比较与torch.nn.utils.clip_grad_value_的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/clip_by_value.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.utils.clip_grad_value_

```text
torch.nn.utils.clip_grad_value_(parameters, clip_value)
```

更多内容详见[torch.nn.utils.clip_grad_value_](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.clip_grad_value_.html)。

## mindspore.ops.clip_by_value

```text
mindspore.ops.clip_by_value(x, clip_value_min=None, clip_value_max=None)
```

更多内容详见[mindspore.ops.clip_by_value](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.clip_by_value.html)。

## 差异对比

PyTorch中梯度是Tensor的属性，可以通过设置`requires_grad=True`使Tensor带有梯度。由于框架机制的不同，在MindSpore中，梯度和权重是互相独立的Tensor。因此在梯度裁剪时，MindSpore需要先获取梯度Tensor再进行裁剪。

PyTorch：实现梯度裁剪可以直接传入带有梯度的Tensor。

MindSpore：由于框架机制不同，实现梯度裁剪，需要先获取梯度，再对梯度进行裁剪。可以使用`mindspore.grad`等方法获取梯度，详情请参考[梯度求导](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/gradient.html#梯度求导)

| 分类 |  子类  |   PyTorch   | MindSpore | 差异                 |
| ---- | ----- | ----------- | --------- | -------------------- |
| 参数 | 参数1 | parameters   | x        | 梯度机制不同，PyTorch传入Tensor即可对梯度裁剪，MindSpore需要传入梯度Tensor，如何获取梯度请参考[梯度求导](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/gradient.html#梯度求导)。 |
|      | 参数2 | clip_value   | clip_value_min        | PyTorch裁剪范围为[-clip_value, clip_value]，MindSpore裁剪范围为[clip_value_min, clip_value_max] |
|      | 参数3 | -            | clip_value_max   |  PyTorch裁剪范围为[-clip_value, clip_value]，MindSpore裁剪范围为[clip_value_min, clip_value_max] |

### 代码示例

> 由于机制不同，MindSpore实现梯度裁剪需要先使用[mindspore.grad](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.grad.html)等方法获取梯度（更多获取梯度的方法，请参考[梯度求导](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/model_development/gradient.html#梯度求导)），再对梯度进行裁剪，示例代码如下。

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
