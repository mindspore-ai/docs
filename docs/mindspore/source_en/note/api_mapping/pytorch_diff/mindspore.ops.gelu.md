# Differences with torch.nn.functional.gelu

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/docs/mindspore/source_en/note/api_mapping/pytorch_diff/mindspore.ops.gelu.md)

## torch.nn.functional.gelu

```text
torch.nn.functional.gelu(input) -> Tensor
```

For more information, see [torch.nn.functional.gelu](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.gelu).

## mindspore.ops.gelu

```text
mindspore.ops.gelu(input_x, approximate='none')
```

For more information, see [mindspore.ops.gelu](https://www.mindspore.cn/docs/en/r2.2/api_python/ops/mindspore.ops.gelu.html).

## Differences

PyTorch: This function represents the Gaussian error linear unit function $GELU(X)=X\times \Phi(x)$, where $\Phi(x)$ is the cumulative distribution function of the Gaussian distribution. The input x denotes an arbitrary number of dimensions.

MindSpore:  MindSpore API implements basically the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameter | Parameter 1 |    -     | approximate | There are two gelu approximation algorithms: 'none' and 'tanh', and the default value is 'none'. After testing, the output is more similar to Pytorch when approximate is 'none'. |
| Input | Single input| input      | input_x           | Same function, different parameter names               |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
input = torch.Tensor([[2, 4], [1, 2]])
output = torch.nn.functional.gelu(input)
print(output.detach().numpy())
# [[1.9544997 3.9998734]
#  [0.8413447 1.9544997]]

# MindSpore
import mindspore
import numpy as np
x = mindspore.Tensor(np.array([[2, 4], [1, 2]]), mindspore.float32)
output = mindspore.ops.gelu(x)
print(output)
# [[1.9545997 3.99993]
#  [0.841192 1.9545977]]
```
