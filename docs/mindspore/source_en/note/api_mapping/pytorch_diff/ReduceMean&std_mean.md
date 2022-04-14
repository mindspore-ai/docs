# Function Differences with torch.std_mean

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ReduceMean&std_mean.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

## torch.std_mean

```python
torch.std_mean(
    input,
    dim,
    unbiased=True,
    keepdim=False
)
```

For more information, see [torch.std_mean](https://pytorch.org/docs/1.5.0/torch.html#torch.std_mean).

## mindspore.ops.ReduceMean

```python
class mindspore.ops.ReduceMean(keep_dims=False)(
    input_x,
    axis=()
)
```

For more information, see [mindspore.ops.ReduceMean](https://mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.ReduceMean.html#mindspore.ops.ReduceMean).

## Differences

PyTorch: Computes standard-deviation and mean of the given axis.

MindSpore：Computes mean of the given axis.

## Code Example

```python
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, only the mean of given dimension will be returned.
input_x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
op = ops.ReduceMean(keep_dims=True)
output = op(x=input_x, axis=1)
print(output)
# Out：
# [[1.5]
#  [3.5]]

# In torch, both std and mean of given dimensions will be returned.
input_x = torch.tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
output = torch.std_mean(input=input_x, dim=1)
std, mean = output
print('std: {}'.format(std))
print('mean: {}'.format(mean))
# Out：
# torch.tensor([0.7071, 0.7071])
# torch.tensor([1.5000, 3.5000])
```