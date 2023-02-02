# Function Differences with torch.nn.Sigmoid

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Sigmoid.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Sigmoid

```text
class torch.nn.Sigmoid()(input) -> Tensor
```

For more information, see [torch.nn.Sigmoid](https://pytorch.org/docs/1.8.1/generated/torch.nn.Sigmoid.html).

## mindspore.nn.Sigmoid

```text
class mindspore.nn.Sigmoid()(input_x) -> Tensor
```

For more information, see [mindspore.nn.Sigmoid](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Sigmoid.html).

## Differences

PyTorch: Compute Sigmoid activation function element-wise, which maps the input to between 0 and 1.

MindSpore: MindSpore API implements the same functionality as PyTorch, and only the input parameter names after instantiation are different.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| :-: | :-: | :-: | :-: |:-:|
|Input | Single input | input | input_x |Same function, different parameter names |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor

input_x = tensor([-1, -2, 0, 2, 1], dtype=torch.float32)
sigmoid = torch.nn.Sigmoid()
output = sigmoid(input_x).numpy()
print(output)
# [0.26894143 0.11920292 0.5        0.880797   0.7310586 ]

# MindSpore
import mindspore
from mindspore import Tensor

input_x = Tensor([-1, -2, 0, 2, 1], mindspore.float32)
sigmoid = mindspore.nn.Sigmoid()
output = sigmoid(input_x)
print(output)
# [0.26894143 0.11920292 0.5        0.8807971  0.7310586 ]
```
