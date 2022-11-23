# 比较与torch.unsqueeze的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/expand_dims.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.unsqueeze

```text
torch.unsqueeze(input, dim) -> Tensor
```

更多内容详见 [torch.unsqueeze](https://pytorch.org/docs/1.8.1/generated/torch.unsqueeze.html)。

## mindspore.ops.expand_dims

```text
mindspore.ops.expand_dims(input_x, axis) -> Tensor
```

更多内容详见 [mindspore.ops.expand_dims](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.expand_dims.html)。

## 差异对比

PyTorch:对输入input在给定的轴上添加额外维度。

MindSpore:MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类 | 子类  | PyTorch | MindSpore | 差异                  |
| ---- | ----- | ------- | --------- | --------------------- |
| 参数 | 参数1 | input   | input_x   | 功能一致， 参数名不同 |
|      | 参数2 | dim     | axis      | 功能一致， 参数名不同 |

### 代码示例

> 两API实功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor

x = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.float32)
dim = 1
out = torch.unsqueeze(x,dim).numpy()
print(out)
# [[[ 1.  2.  3.  4.]]
#  [[ 5.  6.  7.  8.]]
#  [[ 9. 10. 11. 12.]]]

# MindSpore
import mindspore
import numpy as np
import mindspore.ops as ops
from mindspore import Tensor

input_params = Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), mindspore.float32)
axis = 1
output = ops.expand_dims(input_params,  axis)
print(output)
# [[[ 1.  2.  3.  4.]]
#  [[ 5.  6.  7.  8.]]
#  [[ 9. 10. 11. 12.]]]
```