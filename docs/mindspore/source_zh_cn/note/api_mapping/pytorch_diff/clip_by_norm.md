# 比较与torch.nn.utils.clip_grad_norm_的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/clip_by_norm.md)

## torch.nn.utils.clip_grad_norm_

```text
torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=2.0)
```

更多内容详见[torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/1.8.1/generated/torch.nn.utils.clip_grad_norm_.html)。

## mindspore.ops.clip_by_norm

```text
mindspore.ops.clip_by_norm(x, max_norm, norm_type=2.0, error_if_nonfinite=False)
```

更多内容详见[mindspore.ops.clip_by_norm](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/ops/mindspore.ops.clip_by_norm.html)。

## 差异对比

PyTorch中梯度是Tensor的属性，可以通过设置`requires_grad=True`使Tensor带有梯度。由于框架机制的不同，在MindSpore中，梯度和权重是互相独立的Tensor。因此在梯度裁剪时，MindSpore需要先获取梯度Tensor再进行裁剪。

PyTorch：实现梯度裁剪可以直接传入带有梯度的Tensor。

MindSpore：由于框架机制不同，实现梯度裁剪，需要先获取梯度，再对梯度进行裁剪。可以使用`mindspore.grad`等方法获取梯度，详情请参考[梯度求导](https://www.mindspore.cn/docs/zh-CN/r2.2/migration_guide/model_development/gradient.html#梯度求导)。

| 分类 |  子类  |   PyTorch   | MindSpore | 差异                 |
| ---- | ----- | ----------- | --------- | -------------------- |
| 参数 | 参数1 | parameters   | x        | 梯度机制不同，PyTorch传入Tensor即可对梯度裁剪，MindSpore需要传入梯度Tensor，如何获取梯度请参考[梯度求导](https://www.mindspore.cn/docs/zh-CN/r2.2/migration_guide/model_development/gradient.html#梯度求导)。 |
|      | 参数2 | max_norm   | max_norm  | 一致 |
|      | 参数3 | norm_type  | norm_type | 一致 |
|      | 参数4 | -  | error_if_nonfinite | PyTorch1.12版本新增参数，功能与1.12一致 |

### 代码示例

> 由于机制不同，MindSpore实现梯度裁剪需要先使用[mindspore.grad](https://www.mindspore.cn/docs/zh-CN/r2.2/api_python/mindspore/mindspore.grad.html)等方法获取梯度（更多获取梯度的方法，请参考[梯度求导](https://www.mindspore.cn/docs/zh-CN/r2.2/migration_guide/model_development/gradient.html#梯度求导)），再对梯度进行裁剪，示例代码如下。

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
