# Function Differences with torch.clamp

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/pytorch_diff/clamp.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

> `torch.clip` alias for `torch.clamp`，`torch.Tensor.clip` alias for `torch.Tensor.clamp`.
>
> `mindspore.ops.clip` alias for `mindspore.ops.clamp`，`mindspore.Tensor.clip` alias for `mindspore.Tensor.clamp`.
>
> The functional differences between `mindspore.ops.clip` and `torch.clip`, `mindspore.Tensor.clamp` and `torch.Tensor.clamp`, `mindspore.Tensor.clip` and `torch.Tensor.clip`, can refer to the functional differences between `mindspore.ops.clamp` and `torch.clamp`.

## torch.clamp

```text
torch.clamp(input, min, max, *, out=None) -> Tensor
```

For more information, see [torch.clamp](https://pytorch.org/docs/1.8.1/generated/torch.clamp.html).

## mindspore.ops.clamp

```text
mindspore.ops.clamp(x, min=None, max=None) -> Tensor
```

For more information, see [mindspore.ops.clamp](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/ops/mindspore.ops.clamp.html).

## Differences

PyTorch: Clamps all elements in `input` into the range `[min, max]`. Change the value smaller than `min` to `min` and the value larger than `max` to `max`.

MindSpore: MindSpore API implements the same functionality as PyTorch except for the `input` parameter name. The dtype of the output is the same as the input `x`.

| Categories | Subcategories | PyTorch | MindSpore | Differences       |
| --- |---------------|---------| --- |-------------------------------------------------------------------|
|Parameter | Parameter 1 | input   | x | The function is the same, and the parameter `input` name is different |
| | Parameter 2   | min  | min | The function is the same          |
| | Parameter 3   | max  | max | The function is the same          |
| | Parameter 4   | out  | - | MindSpore does not have this Parameter      |

### Code Example 1

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

### Code Example 2

> MindSpore API returns the same dtype as the input `x`.

```python
# PyTorch
import torch
a = torch.tensor([1, 2, 3])
max = torch.tensor([0.5, 0.6, 0.7])
output = torch.clamp(a, max=max)
print(output)
#tensor([0.5000, 0.6000, 0.7000])

# MindSpore
x = Tensor([1., 2., 3.])
max = Tensor([0.5, 0.6, 0.7])
output = ops.clamp(x, max=max)
print(output)
#[0.5 0.6 0.7]
x = Tensor([1, 2, 3])
max = Tensor([0.5, 0.6, 0.7])
output = ops.clamp(x, max=max)
print(output)
#[0 0 0]
```
