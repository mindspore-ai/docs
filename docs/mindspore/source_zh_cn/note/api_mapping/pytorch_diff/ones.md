# 比较与torch.ones的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ones.md)

## torch.ones

```text
torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
```

更多内容详见[torch.ones](https://pytorch.org/docs/1.8.1/generated/torch.ones.html)。

## mindspore.ops.ones

```text
mindspore.ops.ones(shape, dtype=dtype) -> Tensor
```

更多内容详见[mindspore.ops.ones](https://mindspore.cn/docs/zh-CN/r2.0/api_python/ops/mindspore.ops.ones.html)。

## 差异对比

PyTorch：生成大小为 `*size` 的填充值为1的Tensor。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类  | 子类  | PyTorch       | MindSpore | 差异                         |
|-----|-----|---------------|-----------|----------------------------|
| 参数  | 参数1 | size          | shape     | MindSpore支持int、tuple或Tensor类型的输入 |
|     | 参数2 | out           | -         | 不涉及                        |
|     | 参数3 | dtype         | dtype     | 无差异                        |
|     | 参数4 | layout        | -         | 不涉及                        |
|     | 参数5 | device        | -         | 不涉及                        |
|     | 参数6 | requires_grad | -         | 不涉及                        |

### 代码示例1

两API实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor

output = torch.ones(2, 2, dtype=torch.float32)
print(output.numpy())
# [[1. 1.]
#  [1. 1.]]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor

output = ops.ones((2, 2), dtype=ms.float32).asnumpy()
print(output)
# [[1. 1.]
#  [1. 1.]]
```

