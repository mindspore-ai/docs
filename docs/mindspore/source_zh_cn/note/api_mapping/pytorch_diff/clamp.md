# 比较与torch.clamp的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/clamp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

> `torch.clip`别名`torch.clamp`，`torch.Tensor.clip`别名`torch.Tensor.clamp`。
>
> `mindspore.ops.clip`别名`mindspore.ops.clamp`，`mindspore.Tensor.clip`别名`mindspore.Tensor.clamp`。
>
> `mindspore.ops.clip`与`torch.clip`，`mindspore.Tensor.clamp`与`torch.Tensor.clamp`,`mindspore.Tensor.clip`与`torch.Tensor.clip`的功能差异，均参考`mindspore.ops.clamp`与`torch.clamp`的功能差异比较。

## torch.clamp

```text
torch.clamp(input, min, max, *, out=None) -> Tensor
```

更多内容详见[torch.clamp](https://pytorch.org/docs/1.8.1/generated/torch.clamp.html)。

## mindspore.ops.clamp

```text
mindspore.ops.clamp(x, min=None, max=None) -> Tensor
```

更多内容详见[mindspore.ops.clamp](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.clamp.html)。

## 差异对比

PyTorch：将`input`中所有元素限制在`[min, max]`范围中，将比`min`更小的值变为`min`，比`max`更大的值变为`max`。

MindSpore：MindSpore此API实现基本功能与PyTorch一致，参数名`input`不同。返回结果的数据类型和输入`x`相同。

| 分类 | 子类 |PyTorch | MindSpore | 差异         |
| --- | --- | --- | --- |------------|
|参数 | 参数1 | input | x   | 功能一致，参数名不同 |
| | 参数2 | min | min | 功能一致       |
| | 参数3 | max | max | 功能一致       |
| | 参数4 | out | - | 不涉及        |

### 代码示例1

> 两API实现基本功能一致，用法相同。

```python
# PyTorch
import torch
a = torch.tensor([[1., 25., 5., 7.], [4., 11., 6., 21.]], dtype=torch.float32)
x = torch.clamp(a, min=5, max=20)
print(x.detach().numpy())
#[[ 5. 20.  5.  7.]
# [ 5. 11.  6. 20.]]

# MindSpore
import mindspore
from mindspore import Tensor, ops
import numpy as np
min = Tensor(5, mindspore.float32)
max = Tensor(20, mindspore.float32)
x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
output = ops.clamp(x, min, max)
print(output)
#[[ 5. 20.  5.  7.]
# [ 5. 11.  6. 20.]]
```

### 代码示例2

> MindSpore此API返回结果的数据类型和输入`x`相同。

```python
# PyTorch
import torch
a = torch.tensor([1, 2, 3])
max = torch.tensor([0.5, 0.6, 0.7])
output = torch.clamp(a, max=max)
print(output)
#tensor([0.5000, 0.6000, 0.7000])

# MindSpore
x = Tensor([1., 2., 3.])
max = Tensor([0.5, 0.6, 0.7])
output = ops.clamp(x, max=max)
print(output)
#[0.5 0.6 0.7]
x = Tensor([1, 2, 3])
max = Tensor([0.5, 0.6, 0.7])
output = ops.clamp(x, max=max)
print(output)
#[0 0 0]
```
