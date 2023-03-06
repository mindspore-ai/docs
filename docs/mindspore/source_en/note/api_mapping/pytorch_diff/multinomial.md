# Function Differences with torch.multinomial

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/multinomial.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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
mindspore.ops.multinomial(inputs, num_sample, replacement=True, seed=None)
```

For more information, see [mindspore.ops.multinomial](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.multinomial.html).

## Differences

There are differences in parameter names and default values between MindSpore and PyTorch, but there is no difference in functionality.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | input        | inputs        | The functions are the same, but the parameter names are different |
|            | Parameter 2   | num_samples  | num_sample   | The functions are the same, but the parameter names are different |
|            | Parameter 3   | replacement  | replacement   | The functionality is the same, the default values are different. The default value for PyTorch is False and the default value for MindSpore is True  |
|            | Parameter 4   | generator          | seed    | MindSpore uses a random number seed to generate random numbers |
|            | Parameter 5   | out          | -             | Not involved  |

## Code Example

```python
# PyTorch
import torch

x = torch.tensor([0, 9, 4, 0], dtype=torch.float32)
output = torch.multinomial(x, 2)
print(output)
# tensor([1, 2]) or tensor([2, 1])

# MindSpore
import mindspore as ms

x = ms.Tensor([0, 9, 4, 0], dtype=ms.float32)
output = ms.ops.multinomial(x, 2, False)
print(output)
# [1 2] or [2 1]
```
