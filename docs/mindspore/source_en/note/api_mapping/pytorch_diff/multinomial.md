# Function Differences with torch.multinomial

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/multinomial.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

The following mapping relationships can be found in this file.

|     PyTorch APIs          |      MindSpore APIs           |
| :-------------------:     | :-----------------------:     |
| torch.multinomial         | mindspore.ops.multinomial     |
| torch.Tensor.multinomial  | mindspore.Tensor.multinomial  |

## torch.multinomial

```python
torch.multinomial(input, num_samples, replacement=False, *, generator=None, out=None)
```

For more information, see [torch.multinomial](https://pytorch.org/docs/1.8.1/generated/torch.multinomial.html).

## mindspore.ops.multinomial

```python
mindspore.ops.multinomial(input, num_samples, replacement=True, seed=None)
```

For more information, see [mindspore.ops.multinomial](https://www.mindspore.cn/docs/en/r2.0/api_python/ops/mindspore.ops.multinomial.html).

## Differences

There are differences in parameter names and default values between MindSpore and PyTorch, but there is no difference in functionality.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | input        | input         | Consistent    |
|            | Parameter 2   | num_samples  | num_samples   | Consistent    |
|            | Parameter 3   | replacement  | replacement   | The functionality is the same, the default values are different. The default value for PyTorch is False and the default value for MindSpore is True  |
|            | Parameter 4   | generator    | seed          | MindSpore uses a random number seed to generate random numbers |
|            | Parameter 5   | out          | -             | Not involved  |

## Code Example

```python
# PyTorch
import torch

input = torch.tensor([0, 9, 4, 0], dtype=torch.float32)
output = torch.multinomial(input, 2)
print(output)
# tensor([1, 2]) or tensor([2, 1])

# MindSpore
import mindspore as ms

input = ms.Tensor([0, 9, 4, 0], dtype=ms.float32)
output = ms.ops.multinomial(input, 2, False)
print(output)
# [1 2] or [2 1]
```
