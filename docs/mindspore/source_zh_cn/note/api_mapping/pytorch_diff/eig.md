# 比较与torch.eig的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/eig.md)

## torch.eig

```text
torch.eig(input, eigenvectors=False, *, out=None) -> Tensor
```

更多内容详见[torch.eig](https://pytorch.org/docs/1.8.1/generated/torch.eig.html#torch.eig)。

## mindspore.ops.eig

```text
mindspore.ops.eig(A) -> Tensor
```

更多内容详见[mindspore.ops.eig](https://mindspore.cn/docs/zh-CN/r1.11/api_python/ops/mindspore.ops.eig.html)。

## 差异对比

PyTorch：如果 `eigenvectors`为True，则返回 `eigenvalues` 和 `eigenvectors`；如果为False，则只返回`eigenvalues`。在1.9版本以后 `torch.eig` 已经被 `torch.linalg.eig` 取代，`mindspore.ops.eig` 与 `torch.linalg.eig` 接口一致。

MindSpore：返回 `eigenvalues` 和 `eigenvectors`。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                    |
| ---- | ----- | ------- | --------- | --------------------------------------- |
| 参数 | 参数1 | input   | A         | 功能一致，参数名不同                    |
|      | 参数2 | eigenvectors   | -      |MindSpore无此参数  |
|      | 参数3 | out   | -         | PyTorch的 `out` 可以获取输出，MindSpore无此参数 |

### 代码示例

```python
# PyTorch
import numpy as np
import torch

inputs = np.array([[1.0, 0.0], [0.0, 2.0]])
value, vector = torch.eig(torch.tensor(inputs), eigenvectors=True)
print(value)
# tensor([[1., 0.],
          [2., 0.]], dytpe=torch.float64)
print(vector)
# tensor([[1., 0.],
          [0., 1.]], dytpe=torch.float64)

# MindSpore
import mindspore
from mindspore import ops

value, vector = mindspore.ops.eig(Tensor(inputs, mindspore.float32))
print(value)
# [1.+0.j 2.+0.j]
print(vector)
# [[1.+0.j 0.+0.j]
#  [0.+0.j 1.+0.j]]
```
