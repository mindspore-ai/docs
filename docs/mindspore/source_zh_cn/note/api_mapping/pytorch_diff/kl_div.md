# 比较与torch.nn.functional.kl_div的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/kl_div.md)

## torch.nn.functional.kl_div

```text
torch.nn.functional.kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False)
```

更多内容详见[torch.nn.functional.kl_div](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.kl_div)。

## mindspore.ops.kl_div

```text
mindspore.ops.kl_div(logits, labels, reduction='mean')
```

更多内容详见[mindspore.ops.kl_div](https://mindspore.cn/docs/zh-CN/r2.3/api_python/ops/mindspore.ops.kl_div.html)。

## 差异对比

PyTorch：计算输入 `logits` 和 `labels` 的KL散度， `log_target` 标志 `target` 是否传递到log空间。

MindSpore：MindSpore此API实现功能与PyTorch一致，但未设置 `log_target` 参数。

| 分类 | 子类 | PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 输入1 | input | logits | 都是输入Tensor |
| | 输入2 | target | labels | 都是输入Tensor |
| | 参数2 | size_average | - | 功能一致，PyTorch已弃用该参数，MindSpore无此参数 |
| | 参数3 | reduce | - | 功能一致，PyTorch已弃用该参数，MindSpore无此参数 |
| | 参数4 | reduction | reduction | 功能一致，参数名相同 |
| | 参数5 | log_target | - | 参数未设定 |

### 代码示例

```python
# PyTorch
import torch
import numpy as np

logits = torch.tensor(np.array([0.2, 0.7, 0.1]))
labels = torch.tensor(np.array([0., 1., 0.]))
output = torch.nn.functional.kl_div(logits, labels)
print(output)
# tensor(-0.2333, dtype=torch.float64)

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

logits = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
labels = Tensor(np.array([0., 1., 0.]), mindspore.float32)
output = mindspore.ops.kl_div(logits, labels, 'mean')
print(output)
# -0.23333333
```
