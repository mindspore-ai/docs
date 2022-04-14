# Function Differences with torch.nn.init.uniform_

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Uniform.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.init.uniform_

```python
torch.nn.init.uniform_(
    tensor,
    a=0.0,
    b=1.0
)
```

For more information, see [torch.nn.init.uniform_](https://pytorch.org/docs/1.5.0/nn.init.html#torch.nn.init.uniform_).

## mindspore.common.initializer.Uniform

```python
class mindspore.common.initializer.Uniform(scale=0.07)(arr)
```

For more information, see [mindspore.common.initializer.Uniform](https://mindspore.cn/docs/en/r1.7/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Uniform).

## Differences

PyTorch: The upper and lower bounds of uniform distribution are specified by parameters `a` and `b`, i.e. U(-a, b).

MindSpore：It only uses one parameter to specify a uniformly distributed range, i.e. U(-scale, scale) and update-in-place for the input.

## Code Example

```python
import mindspore
import torch
import numpy as np

# In MindSpore, only one parameter is set to specify the scope of uniform distribution (-1, 1).
input_x = np.array([1, 1, 1]).astype(np.float32)
uniform = mindspore.common.initializer.Uniform(scale=1)
uniform(input_x)
print(input_x)
# Out：
# [-0.2333 0.6208 -0.1627]

# In torch, parameters are set separately to specify the lower and upper bound of uniform distribution.
input_x = torch.tensor(np.array([1, 1, 1]).astype(np.float32))
output = torch.nn.init.uniform_(tensor=input_x, a=-1, b=1)
print(output)
# Out：
# tensor([0.9936, 0.7676, -0.8275])
```