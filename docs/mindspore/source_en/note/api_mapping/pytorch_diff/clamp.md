# Function Differences with torch.clamp

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/clamp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.clamp

```text
torch.clamp(input, min, max, *, out=None) -> Tensor
```

For more information, see [torch.clamp](https://pytorch.org/docs/1.8.1/generated/torch.clamp.html).

## mindspore.ops.clamp

```text
mindspore.ops.clamp(x, min=None, max=None) -> Tensor
```

For more information, see [mindspore.ops.clamp](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.clamp.html).

## Differences

PyTorch: Clamps all elements in `input` into the range `[min, max]`. Change the value smaller than `min` to `min` and the value larger than `max` to `max`.

MindSpore: MindSpore API implements the same functionality as PyTorch except for the `input` parameter name.

| Categories | Subcategories | PyTorch | MindSpore | Differences       |
| --- |---------------|---------| --- |-------------------------------------------------------------------|
|Parameter | Parameter 1 | input   | x | The function is the same, and the parameter `input` name is different |
| | Parameter 2   | min  | min | The function is the same          |
| | Parameter 3   | max  | max | The function is the same          |
| | Parameter 4   | out  | - | MindSpore does not have this Parameter      |

## Code Example 1

> The two APIs have the same functions and the same usage.

```python
# PyTorch
import torch
a = torch.tensor([[1., 25., 5., 7.], [4., 11., 6., 21.]], dtype=torch.float32)
x = torch.clamp(a, min=5, max=20)
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
output = ops.clamp(x, min, max)
print(output)
#[[ 5. 20.  5.  7.]
# [ 5. 11.  6. 20.]]
```
