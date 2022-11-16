# 比较与torch.erfinv的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/erfinv.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.erfinv

```text
torch.erfinv(input, *, out=None) -> Tensor
```

更多内容详见 [torch.erfinv](https://pytorch.org/docs/1.8.1/generated/torch.erfinv.html)。

## mindspore.ops.erfinv

```text
mindspore.ops.erfinv(input) -> Tensor
```

更多内容详见 [mindspore.ops.erfinv](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.erfinv.html)。

## 差异对比

PyTorch：计算输入Tensor的逆误差函数。逆误差函数在范围(-1, 1)间，公式为：$erfinv(erf(x))=x$。

MindSpore：MindSpore此API实现功能与PyTorch一致。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | input |- |
| | 参数2 | out | - |不涉及 |

### 代码示例

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor

input_x = tensor([0, 0.5, -0.9], dtype=torch.float32)
out = torch.erfinv(input_x).numpy()
print(out)
# [ 0.          0.47693628 -1.1630871 ]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

x = Tensor(np.array([0, 0.5, -0.9]), mindspore.float32)
output = mindspore.ops.erfinv(x)
print(output)
# [ 0.          0.47693628 -1.163087  ]
```
