# Function Differences with torch.nn.functional.normalize

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_en/note/api_mapping/pytorch_diff/L2Normalize.md)

## torch.nn.functional.normalize

```python
torch.nn.functional.normalize(
    input,
    p=2,
    dim=1,
    eps=1e-12,
    out=None
)
```

For more information, see [torch.nn.functional.normalize](https://pytorch.org/docs/1.5.0/nn.functional.html#torch.nn.functional.normalize).

## mindspore.ops.L2Normalize

```python
class mindspore.ops.L2Normalize(
    axis=0,
    epsilon=1e-4
)(input_x)
```

For more information, see [mindspore.ops.L2Normalize](https://mindspore.cn/docs/en/r1.11/api_python/ops/mindspore.ops.L2Normalize.html#mindspore.ops.L2Normalize).

## Differences

PyTorch: Supports using the LP paradigm by specifying the parameter `p`. The calculation formula is input as the numerator, and the sum of the squares of the input is taken as the square root first, and then the max corresponding to epsilon is obtained as the denominator. The function definition is as follows:

$$
v = \frac{v}{\max(\lVert v \rVert_p, \epsilon)}.
$$

MindSpore: Only L2 paradigm is supported. The calculation formula is input as the numerator, and the sum of the squares of the input first finds the max corresponding to epsilon, and then takes the square root as the denominator. The function definition is as follows:

$$
\displaylines{{\text{output} = \frac{x}{\sqrt{\text{max}( \sum_{i}^{}\left | x_i  \right | ^2, \epsilon)}}}}
$$

Due to differences in calculation formulas, there are differences in calculation results under certain inputs.

### Code Example 1

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, you can directly pass data into the function, and the default dimension is 0.
l2_normalize = ops.L2Normalize()
input_x = ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
output = l2_normalize(input_x)
print(output)
# [0.2673 0.5345 0.8018]

# In torch, parameter p should be set to determine it is a lp normalization, and the default dimension is 1.
input_x = torch.tensor(np.array([1.0, 2.0, 3.0]))
outputL2 = torch.nn.functional.normalize(input=input_x, p=2, dim=0)
outputL3 = torch.nn.functional.normalize(input=input_x, p=3, dim=0)
print(outputL2)
# tensor([0.2673, 0.5345, 0.8018], dtype=torch.float64)
print(outputL3)
# tensor([0.3029, 0.6057, 0.9086], dtype=torch.float64)
```

### Code Example 2

> There are differences in the calculation formulas of the two APIs. When the input data is particularly small, there will be huge differences in results.

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, you can directly pass data into the function, and the default dimension is 0.
l2_normalize = ops.L2Normalize()
input_x = ms.Tensor(np.array([1.0 * 1e-10, 2.0 * 1e-10, 3.0 * 1e-10]), ms.float32)
output = l2_normalize(input_x)
print(output)
# [1.e-08 2.e-08 3.e-08]

# In torch, parameter p should be set to determine it is a lp normalization, and the default dimension is 1.
input_x = torch.tensor(np.array([1.0 * 1e-10, 2.0 * 1e-10, 3.0 * 1e-10]))
outputL2 = torch.nn.functional.normalize(input=input_x, p=2, dim=0)
outputL3 = torch.nn.functional.normalize(input=input_x, p=3, dim=0)
print(outputL2)
# tensor([0.2673, 0.5345, 0.8018], dtype=torch.float64)
print(outputL3)
# tensor([0.3029, 0.6057, 0.9086], dtype=torch.float64)
```