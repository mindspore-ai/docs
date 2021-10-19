# Function Differences with torch.nn.Flatten

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/nn_Flatten.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## torch.nn.Flatten

```python
class torch.nn.Flatten(
    start_dim=1,
    end_dim=-1
)
```

For more information, see [torch.nn.Flatten](https://pytorch.org/docs/1.5.0/nn.html#torch.nn.Flatten).

## mindspore.nn.Flatten

```python
class mindspore.nn.Flatten()(input)
```

For more information, see [mindspore.nn.Flatten](https://mindspore.cn/docs/api/en/r1.5/api_python/nn/mindspore.nn.Flatten.html#mindspore.nn.Flatten).

## Differences

PyTorch: Supports the flatten of elements by specified dimensions. This should be used together with `torch.nn.Sequential`

MindSpore：Only the 0th dimension element is reserved and the elements of the remaining dimensions are flattened.

## Code Example

```python
import mindspore
from mindspore import Tensor, nn
import torch
import numpy as np

# In MindSpore, only the 0th dimension will be reserved and the rest will be flattened.
input_tensor = Tensor(np.ones(shape=[1, 2, 3, 4]), mindspore.float32)
flatten = nn.Flatten()
output = flatten(input_tensor)
print(output.shape)
# Out：
# (1, 24)

# In torch, the dimension to reserve can be specified and the rest will be flattened.
# Different from torch.flatten, you should pass it as parameter into torch.nn.Sequential.
input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
flatten1 = torch.nn.Sequential(torch.nn.Flatten(start_dim=1))
output1 = flatten1(input_tensor)
print(output1.shape)
# Out：
# torch.Size([1, 24])

input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
flatten2 = torch.nn.Sequential(torch.nn.Flatten(start_dim=2))
output2 = flatten2(input_tensor)
print(output2.shape)
# Out：
# torch.Size([1, 2, 12])
```