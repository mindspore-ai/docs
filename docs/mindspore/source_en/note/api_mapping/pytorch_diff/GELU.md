# Function Differences with torch.nn.GELU

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/GELU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## torch.nn.GELU

```python
class torch.nn.GELU()(input) -> Tensor
```

For more information, see [torch.nn.GELU](https://pytorch.org/docs/1.8.1/generated/torch.nn.GELU.html).

## mindspore.nn.GELU

```python
class mindspore.nn.GELU(approximate=True)(x) -> Tensor
```

For more information, see [mindspore.nn.GELU](https://www.mindspore.cn/docs/en/r2.0/api_python/nn/mindspore.nn.GELU.html).

## Differences

PyTorch: This function represents the Gaussian error linear unit function $GELU(X)=X\times \Phi(x)$, where $\Phi(x)$ is the cumulative distribution function of the Gaussian distribution. The input x denotes an arbitrary number of dimensions.

MindSpore:  MindSpore API implements basically the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameter | Parameter 1 |    -     | approximate | Determines whether approximation is enabled or not, and the default value is True. After testing, the output is more similar to Pytorch when approximate is False |
| Input | Single input| input      | x           | Same function, different parameter names               |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
input_x = torch.Tensor([[2, 4], [1, 2]])
output = torch.nn.GELU()(input_x)
print(output.detach().numpy())
# [[1.9544997 3.9998734]
#  [0.8413447 1.9544997]]

# MindSpore
import mindspore
import numpy as np
x = mindspore.Tensor(np.array([[2, 4], [1, 2]]), mindspore.float32)
output = mindspore.nn.GELU(approximate=False)(x)
print(output)
# [[1.9544997 3.9998732]
#  [0.8413447 1.9544997]]
```
