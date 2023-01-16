# Function Differences with torch.nn.GELU

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/GELU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.GELU

```python
class torch.nn.GELU()(input) -> Tensor
```

For more information, see [torch.nn.GELU](https://pytorch.org/docs/1.8.1/generated/torch.nn.GELU.html).

## mindspore.nn.GELU

```python
class mindspore.nn.GELU(approximate=True)(x) -> Tensor
```

For more information, see [mindspore.nn.GELU](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.GELU.html).

## Differences

PyTorch: This function represents the Gaussian error linear unit function $GELU(X)=X\times \Phi(x)$, where $\Phi(x)$ is the cumulative distribution function of the Gaussian distribution. The input x denotes an arbitrary number of dimensions.

MindSpore:  MindSpore API implements basically the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | input      | x           | Interface input, same function, only different parameter names               |
|      | Parameter 2 |    -     | approximate | Determines whether approximation is enabled or not, and the default value is True. If the value of approximimate is True, the Gaussian error linear activation function is: $0.5\times x\times (1+tanh(aqrt(2/pi)\times (x+0.044715 \times x^{3})))$, otherwise it is: $x \times P(X\leqslant x)= 0.5\times x \times (1+erf(x/sqrt(2)))$, where $P(X)\sim N(0,1)$. |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
input = torch.Tensor([[2, 4], [1, 2]])
output = torch.nn.GELU()(input)
print(output.detach().numpy())
# [[1.9544997 3.9998734]
#  [0.8413447 1.9544997]]

# MindSpore
import mindspore
import numpy as np
x = mindspore.Tensor(np.array([[2, 4], [1, 2]]), mindspore.float32)
output = mindspore.nn.GELU()(x)
print(output)
# [[1.9545977 3.99993 ]
#  [0.841192 1.9545977]]
```
