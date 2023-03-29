# Function Differences with torch.nn.functional.kl_div

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/kl_div.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## torch.nn.functional.kl_div

```text
torch.nn.functional.kl_div(input, target, size_average=None, reduce=None, reduction='mean', log_target=False)
```

For more information, see [torch.nn.functional.kl_div](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.kl_div).

## mindspore.ops.kl_div

```text
mindspore.ops.kl_div(logits, labels, reduction='mean')
```

For more information, see [mindspore.ops.kl_div](https://www.mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.kl_div.html#mindspore.ops.kl_div).

## Differnnces

PyTorch: Compute the KL divergence of  `logits` 和 `labels`, `log_target` is the flag Indicates whether the `target` is passed to the log space.

MindSpore: MindSpore API basically implements the same function as PyTorch, but the `log_target` is not defined.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | input | logits  | Same function, different parameter names          |
|      | Parameter 2 | target | labels | Same function, different parameter names |
|      | Parameter 3 | size_average    | -    | PyTorch deprecated parameters, functionally replaced by the reduction parameter          |
|      | Parameter 4 | reduce    | -    | PyTorch deprecated parameters, functionally replaced by the reduction parameter        |
|      | Parameter 5 | reduction | reduction | Same function, different default values. |
|      | Parameter 6| log_target    | -    | parameter not defined    |

### Code Example

```python
import numpy as np
import mindspore
from mindspore import Tensor
import torch

logits = Tensor(np.array([0.2, 0.7, 0.1]), mindspore.float32)
labels = Tensor(np.array([0., 1., 0.]), mindspore.float32)
output = mindspore.ops.kl_div(logits, labels, 'mean')
print(output)
# -0.23333333

logits = torch.tensor(np.array([0.2, 0.7, 0.1]))
labels = torch.tensor(np.array([0., 1., 0.]))
output = torch.nn.functional.kl_div(logits, labels)
print(output)
# -0.23333333
```
