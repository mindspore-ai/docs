# 比较与torch.zeros的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/zeros.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.zeros

```text
torch.zeros(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
```

更多内容详见[torch.zeros](https://pytorch.org/docs/1.8.1/generated/torch.zeros.html)。

## mindspore.ops.zeros

```text
mindspore.ops.zeros(size, dtype=dtype) -> Tensor
```

更多内容详见[mindspore.ops.zeros](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.zeros.html)。

## 差异对比

PyTorch：生成大小为 `*size` 的填充值为0的Tensor。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类  | 子类  | PyTorch       | MindSpore | 差异                         |
|-----|-----|---------------|-----------|----------------------------|
| 参数  | 参数1 | size          | size      | MindSpore只支持int或tuple类型的输入 |
|     | 参数2 | out           | -         | 不涉及                        |
|     | 参数3 | layout        | -         | 不涉及                        |
|     | 参数4 | device        | -         | 不涉及                        |
|     | 参数5 | requires_grad | -         | 不涉及                        |

### 代码示例1

两API实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor

output = torch.zeros(2, 2, dtype=torch.float32)
print(output.numpy())
# [[0. 0.]
#  [0. 0.]]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor

output = ops.zeros((2, 2), dtype=ms.float32).asnumpy()
print(output)
# [[0. 0.]
#  [0. 0.]]
```

