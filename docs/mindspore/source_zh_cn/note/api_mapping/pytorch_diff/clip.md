# 比较与torch.clip的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/clip.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.clip

```text
torch.clip(input, min, max, *, out=None) -> Tensor
```

更多内容详见 [torch.clip](https://pytorch.org/docs/1.8.1/generated/torch.clip.html)。

## mindspore.ops.clip

```text
mindspore.ops.clip(x, min=None, max=None) -> Tensor
```

更多内容详见 [mindspore.ops.clip](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.clip.html)。

## 差异对比

PyTorch：将`input`中所有元素限制在`[min, max]`范围中，将比`min`更小的值变为`min`，比`max`更大的值变为`max`。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名`input`不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异         |
| --- | --- | --- | --- |------------|
|参数 | 参数1 | input | x   | 功能一致，参数名不同 |
| | 参数2 | min | min | 功能一致       |
| | 参数3 | max | max | 功能一致       |
| | 参数4 | out | - | 不涉及        |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
a = torch.tensor([[1., 25., 5., 7.], [4., 11., 6., 21.]], dtype=torch.float32)
x = torch.clip(a, min=5, max=20)
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
output = ops.clip(x, min, max)
print(output)
#[[ 5. 20.  5.  7.]
# [ 5. 11.  6. 20.]]
```
