# Function Differences with torch.flatten

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Flatten.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.flatten

```python
torch.flatten(
    input,
    start_dim=0,
    end_dim=-1
)
```

For more information, see [torch.flatten](https://pytorch.org/docs/1.5.0/torch.html#torch.flatten).

## mindspore.ops.Flatten

```python
class mindspore.ops.Flatten(*args, **kwargs)(input_x)
```

For more information, see [mindspore.ops.Flatten](https://mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.Flatten.html#mindspore.ops.Flatten).

## Differences

PyTorch: Supports the flatten of elements by specified dimensions.

MindSpore：Only the 0th dimension element is reserved and the elements of the remaining dimensions are flattened.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, only the 0th dimension will be reserved and the rest will be flattened.
input_tensor = Tensor(np.ones(shape=[1, 2, 3, 4]), mindspore.float32)
flatten = ops.Flatten()
output = flatten(input_tensor)
print(output.shape)
# Out：
# (1, 24)

# In torch, the dimension to reserve will be specified and the rest will be flattened.
input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
output1 = torch.flatten(input=input_tensor, start_dim=1)
print(output1.shape)
# Out：
# torch.Size([1, 24])

input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
output2 = torch.flatten(input=input_tensor, start_dim=2)
print(output2.shape)
# Out：
# torch.Size([1, 2, 12])
```