# Function Differences with torch.nn.init.constant_

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Constant.md)

## torch.nn.init.constant_

```python
torch.nn.init.constant_(
    tensor,
    val
)
```

For more information, see [torch.nn.init.constant_](https://pytorch.org/docs/1.5.0/nn.init.html#torch.nn.init.constant_).

## mindspore.common.initializer.Constant

```python
class mindspore.common.initializer.Constant(value)(arr)
```

For more information, see [mindspore.common.initializer.Constant](https://mindspore.cn/docs/en/r1.9/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Constant).

## Differences

PyTorch: Fills in the input tensor with constant `val`.

MindSpore：Fills in a constant array with `value`(int or numpy array) and update-in-place for the input.

## Code Example

```python
import mindspore
import torch
import numpy as np

# In MindSpore, fill a constant array with value(int or numpy array).
input_constant = np.array([1, 2, 3])
constant_init = mindspore.common.initializer.Constant(value=1)
out_constant = constant_init(input_constant)
print(input_constant)
# Out：
# [1 1 1]

# In torch, fill in the input tensor with constant val.
input_constant = np.array([1, 2, 3])
out_constant = torch.nn.init.constant_(
    tensor=torch.tensor(input_constant),
    val=1
)
print(out_constant)
# Out：
# tensor([1, 1, 1])
```