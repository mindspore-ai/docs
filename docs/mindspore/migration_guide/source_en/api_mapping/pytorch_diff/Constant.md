# Function Differences with torch.nn.init.constant_

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/Constant.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [mindspore.common.initializer.Constant](https://mindspore.cn/docs/api/en/master/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Constant).

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
# tensor([1., 1., 1.])
```