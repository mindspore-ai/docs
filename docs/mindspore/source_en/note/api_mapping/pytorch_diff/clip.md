# Function Differences with torch.clip

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/clip.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|    torch.clip     |  mindspore.ops.clip   |
|  torch.Tensor.clip | mindspore.Tensor.clip |

## torch.clip

```text
torch.clip(input, min, max, *, out=None) -> Tensor
```

For more information, see [torch.clip](https://pytorch.org/docs/1.8.1/generated/torch.clip.html).

## mindspore.ops.clip

```text
mindspore.ops.clip(x, min=None, max=None) -> Tensor
```

For more information, see [mindspore.ops.clip](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.clip.html).

## Differences

PyTorch: Clamps all elements in `input` into the range `[min, max]`. Change the value smaller than `min` to `min` and the value larger than `max` to `max`.

MindSpore: MindSpore API implements the same functionality as PyTorch except for the `input` parameter name.

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
x = torch.clip(a, min=5, max=20)
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
output = ops.clip(x, min, max)
print(output)
#[[ 5. 20.  5.  7.]
# [ 5. 11.  6. 20.]]
```

### Code Example 2

> In PyTorch-1.8, `min` and `max` only support input of type `Number`, and MindSpore additionally supports Tensor input. Note: When using PyTorch-1.12 to run the first use case of the following code, output `Tensor ([0.5000, 0.5000, 0.5000])`.

```python
# PyTorch
import torch
a = torch.tensor([1, 2, 3])
output = torch.clip(a, max=0.5)
print(output)
#tensor([0, 0, 0])

a = torch.tensor([1., 2., 3.])
output = torch.clip(a, max=0.5)
print(output)
#tensor([0.5000, 0.5000, 0.5000])

# MindSpore
from mindspore import Tensor, ops
x = Tensor([1, 2, 3])
max = Tensor([0.5, 0.5, 0.5])
output = ops.clip(x, max=max)
print(output)
#[0 0 0]

x = Tensor([1., 2., 3.])
output = ops.clip(x, max=max)
print(output)
#[0.5 0.5 0.5]
```
