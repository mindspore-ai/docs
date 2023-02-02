# Function Differences with torch.nn.LeakyReLU

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/LeakyReLU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.LeakyReLU

```text
class torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)(input) -> Tensor
```

For more information, see [torch.nn.LeakyReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.LeakyReLU.html).

## mindspore.nn.LeakyReLU

```text
class mindspore.nn.LeakyReLU(alpha=0.2)(x) -> Tensor
```

For more information, see [mindspore.nn.LeakyReLU](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.LeakyReLU.html).

## Differences

PyTorch: Implements the basic functions of the Leaky ReLU activation function, where the parameter `negative_slope` is used to control the slope of the activation function, and the parameter `inplace` is used to control whether to choose to perform the operation of the activation function in-place.

MindSpore: MindSpore API basically implements the same function as PyTorch, where the parameter `alpha` is the same as the parameter `negative_slope` in PyTorch, with different parameter names and different default values. However, MindSpore does not have the `inplace` parameter.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | negative_slope | alpha |Same function, different parameter names, different default values |
| | Parameter 2 | inplace | - | This parameter is used in PyTorch to control whether to choose to perform the activation function in-place. MindSpore does not have this parameter|
|Input | Single input | input | x | Same function, different parameter names|

### Code Example 1

> PyTorch parameter `negative_slope` and MindSpore parameter `alpha` have the same function, with different parameter names and different default values, and get the same result when both values are the same.

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.LeakyReLU(0.2)
inputs = torch.tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]], dtype=float)
output = m(inputs).to(torch.float32).detach().numpy()
print(output)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

x = Tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype('float32')
m = nn.LeakyReLU()
output = m(x)
print(output)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]
```

### Code Example 2

> PyTorch parameter `inplace` is used to control whether to perform the activation function operation in-place, that is, to perform the activation function operation directly on the input, where the input is changed. MindSpore does not have this parameter, but can assign the output to the input to achieve similar function.

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.LeakyReLU(0.2, inplace=True)
inputs = torch.tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]], dtype=torch.float32)
output = m(inputs)
print(inputs.detach().numpy())
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

x = Tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype('float32')
m = nn.LeakyReLU()
x = m(x)
print(x)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]
```
