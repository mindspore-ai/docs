# Function Differences with torch.bernoulli

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/bernoulli.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

The following mapping relationships can be found in this file.

|     PyTorch APIs          |      MindSpore APIs           |
| :-------------------:     | :-----------------------:     |
| torch.bernoulli           | mindspore.ops.bernoulli       |
| torch.Tensor.bernoulli    | mindspore.Tensor.bernoulli    |

## torch.bernoulli

```python
torch.bernoulli(input, *, generator=None, out=None)
```

For more information, see [torch.bernoulli](https://pytorch.org/docs/1.8.1/generated/torch.bernoulli.html).

## mindspore.ops.bernoulli

```python
mindspore.ops.bernoulli(x, p=0.5, seed=-1)
```

For more information, see [mindspore.ops.bernoulli](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.bernoulli.html).

## Differences

PyTorch: The probability value of the Bernoulli distribution is stored in the parameter `input` , and the shape of the returned value is the same as that of `input` .

MindSpore: The probability value of the Bernoulli distribution is stored in the parameter `p` , with a default value of 0.5. The shape of `p` needs to be consistent with the shape of `x` , and the shape of the return value should be the same as the shape of `x` .

There is no difference in function.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | -            | x             | The shape and data type of the returned value under Mindspore are the same as the shape of `x` |
|            | Parameter 2   | input        | p             | Save the probability values for the Bernoulli distribution. The shape of the return value under PyTorch is the same as 'input'. 'p' is optional under MindSpore, and the default value is 0.5 |
|            | Parameter 3   | generator    | seed          | MindSpore uses a random number seed to generate random numbers |
|            | Parameter 4   | out          | -             | Not involved  |

## Code Example

```python
# PyTorch
import torch
import numpy as np

p0 = np.array([0.0, 1.0, 1.0])
input_torch = torch.tensor(p0, dtype=torch.float32)
output = torch.bernoulli(input_torch)
print(output.shape)
# torch.Size([3])

# MindSpore
import mindspore as ms
import numpy as np

x0 = np.array([1, 2, 3])
p0 = np.array([0.0, 1.0, 1.0])

input_x = ms.Tensor(x0, ms.float32)
input_p = ms.Tensor(p0, ms.float32)
output = ms.ops.bernoulli(input_x, input_p)
print(output.shape)
# (3,)
```