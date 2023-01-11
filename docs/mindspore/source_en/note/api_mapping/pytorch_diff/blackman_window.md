# Function Differences with torch.blackman_window

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/blackman_window.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.blackman_window

```text
torch.blackman_window(
    window_length,
    periodic=True,
    *,
    dtype=None,
    layout=torch.strided,
    device=None,
    requires_grad=False
) -> Tensor
```

For more information, see [torch.blackman_window](https://pytorch.org/docs/1.8.1/generated/torch.blackman_window.html).

## mindspore.ops.blackman_window

```text
mindspore.ops.blackman_window(
    window_length,
    periodic=True,
    *,
    dtype=mstype.float32
) -> Tensor
```

For more information, see [mindspore.ops.blackman_window](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.blackman_window.html).

## Differences

PyTorch: Return a Blackman window with the same size as window_length. The periodic parameter determines whether the returned window will remove the last duplicate value of the symmetric window.

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

torch_output = torch.blackman_window(12, periodic=True)
print(torch_output.numpy())
# [-2.9802322e-08 2.6987284e-02 1.3000000e-01 3.4000000e-01
#  6.3000000e-01 8.9301264e-01 1.0000000e+00 8.9301258e-01
#  6.2999994e-01 3.3999997e-01 1.3000003e-01 2.6987225e-02]

# MindSpore
import mindspore
from mindspore import Tensor

window_length = Tensor(12, mindspore.int32)
ms_output = mindspore.ops.blackman_window(window_length, periodic=True)
print(ms_output.asnumpy())
# [-1.3877788e-17 2.6987297e-02 1.3000000e-01 3.4000000e-01
#  6.3000000e-01 8.9301270e-01 1.0000000e+00 8.9301270e-01
#  6.3000000e-01 3.4000000e-01 1.3000000e-01 2.6987297e-02]
```
