# Function Differences with torch.bartlett_window

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/bartlett_window.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.bartlett_window

```text
torch.bartlett_window(
    window_length,
    periodic=True,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False
) -> Tensor
```

For more information, see [torch.bartlett_window](https://pytorch.org/docs/1.8.1/generated/torch.bartlett_window.html).

## mindspore.ops.bartlett_window

```text
mindspore.ops.bartlett_window(
    window_length,
    periodic=True,
    dtype=mstype.float32
) -> Tensor
```

For more information, see [mindspore.ops.bartlett_window](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.bartlett_window.html).

## Differences

PyTorch: Returns a bartlett window with the same size as window_length. The periodic parameter determines whether the returned window will remove the last duplicate value of the symmetric window.

MindSpore: MindSpore API basically implements the same function as PyTorch, and the precision varies slightly.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 |window_length | window_length | - |
| | Parameter 2 | periodic | periodic | - |
|  | Parameter 3 | dtype        | dtype | - |
| | Parameter 4 | layout | - | Not involved |
| | Parameter 5 | device | - | Not involved |
| | Parameter 6 | requires_grad | - | MindSpore does not have this parameter and supports reverse derivation by default |

### Code Example 1

```python
# PyTorch
import torch

torch_output = torch.bartlett_window(5, periodic=True)
print(torch_output.numpy())
#[0.         0.4        0.8        0.79999995 0.39999998]

# MindSpore
import mindspore
from mindspore import Tensor

window_length = Tensor(5, mindspore.int32)
ms_output = mindspore.ops.bartlett_window(window_length, periodic=True)
print(ms_output.asnumpy())
#[0.  0.4 0.8 0.8 0.4]
```
