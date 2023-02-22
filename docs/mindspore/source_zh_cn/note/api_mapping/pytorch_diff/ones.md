# 比较与torch.ones的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ones.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.ones

```text
torch.ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
```

更多内容详见[torch.ones](https://pytorch.org/docs/1.8.1/generated/torch.ones.html)。

## mindspore.ops.ones

```text
mindspore.ops.ones(size, dtype=dtype) -> Tensor
```

更多内容详见[mindspore.ops.ones](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.ones.html)。

## 差异对比

PyTorch：生成大小为 `*size` 的填充值为1的Tensor。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅参数名不同。

| 分类  | 子类  | PyTorch       | MindSpore | 差异                         |
|-----|-----|---------------|-----------|----------------------------|
| 参数  | 参数1 | size          | size      | MindSpore只支持int或tuple类型的输入 |
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

