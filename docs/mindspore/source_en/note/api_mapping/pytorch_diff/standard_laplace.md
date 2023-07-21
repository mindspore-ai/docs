# Function Differences with torch.distributions.laplace.Laplace

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_en/note/api_mapping/pytorch_diff/standard_laplace.md)

## torch.distributions.laplace.Laplace

```text
torch.distributions.laplace.Laplace(loc, scale) -> Class Instance
```

For more information, see [torch.distributions.laplace.Laplace](https://pytorch.org/docs/1.8.1/distributions.html#torch.distributions.laplace.Laplace).

## mindspore.ops.standard_laplace

```text
mindspore.ops.standard_laplace(shape, seed=None) -> Tensor
```

For more information, see [mindspore.ops.standard_laplace](https://mindspore.cn/docs/en/r1.11/api_python/ops/mindspore.ops.standard_laplace.html).

## Differences

PyTorch: Create a Laplace distribution instance and call the sample interface of the instance to generate random values that match the Laplace distribution.

MindSpore: Generates random numbers that match the standard Laplace (mean=0, lambda=1) distribution. When loc=0, scale=1 in PyTorch and the sample function input shape is the same as MindSpore, the two APIs achieve the same function.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | loc   | -         | MindSpore doesn't have this parameter, it implements the same functionality as loc in PyTorch equals 0   |
|      | Parameter 2 | scale   | -      |  MindSpore doesn't have this parameter, it implements the same functionality as scale in PyTorch equals 1  |
|      | Parameter 3 | -   | shape | This parameter in PyTorch is passed in when the sample interface is called |
|      | Parameter 4 | -   | seed        | Random seeds for the operator layer. PyTorch does not have this parameter |

### Code Example

> Each randomly generated value in PyTorch occupies one dimension, so the innermost layer of the shape passed in MindSpore adds a dimension of length 1, and the two APIs achieve the same function.

```python
# PyTorch
import torch

m = torch.distributions.laplace.Laplace(torch.tensor([0.0]), torch.tensor([1.0]))
shape = (4, 4)
sample = m.sample(shape)
print(tuple(sample.shape))
# (4, 4, 1)

# MindSpore
import mindspore
from mindspore import ops

shape = (4, 4, 1)
output = ops.standard_laplace(shape)
result = output.shape
print(result)
# (4, 4, 1)
```
