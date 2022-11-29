# 比较与torch.clamp的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/clip_by_value.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.clamp

```text
torch.clamp(input, min, max, *, out=None) -> Tensor
```

更多内容详见 [torch.clamp](https://pytorch.org/docs/1.8.1/generated/torch.clamp.html)。

## mindspore.ops.clip_by_value

```text
mindspore.ops.clip_by_value(x, clip_value_min=None, clip_value_max=None) -> Tensor
```

更多内容详见 [mindspore.ops.clip_by_value](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.clip_by_value.html)。

## 差异对比

PyTorch：将input中所有元素限制在[min,max]范围中，将比min更小的值变为min，比max更大的值变为max。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | x              | 功能一致，参数名不同 |
| | 参数2 | min | clip_value_min | 功能一致，参数名不同|
| | 参数3 | max | clip_value_max | 功能一致，参数名不同 |
| | 参数4 | out | - | 不涉及 |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
a = torch.tensor([[1., 25., 5., 7.], [4., 11., 6., 21.]],dtype=torch.float32)
x=torch.clamp(a, min=5, max=20)
print(x.detach().numpy())
#[[ 5. 20.  5.  7.]
# [ 5. 11.  6. 20.]]

# MindSpore
import mindspore
import torch
from mindspore import Tensor, ops
import numpy as np
min = Tensor(5, mindspore.float32)
max = Tensor(20, mindspore.float32)
input = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
output = ops.clip_by_value(input, min, max)
print(output)
# [[ 5. 20.  5.  7.]
#  [ 5. 11.  6. 20.]]
```
