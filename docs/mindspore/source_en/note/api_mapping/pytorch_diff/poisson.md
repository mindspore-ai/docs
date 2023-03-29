# Function Differences with torch.poisson

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/poisson.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## torch.poisson

```python
torch.poisson(input, generator=None)
```

For more information, see [torch.poisson](https://pytorch.org/docs/1.8.1/generated/torch.poisson.html).

## mindspore.ops.random_poisson

```python
mindspore.ops.random_poisson(shape, rate, seed=None, dtype=mstype.float32)
```

For more information, see [mindspore.ops.random_poisson](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.random_poisson.html).

## Differences

PyTorch: The shape and data type of the return value are the same as `input`.

MindSpore: `shape` determines the shape of the random number tensor sampled under each distribution, and the shape of the return value is `mindspore.concat([shape, mindspore.shape(rate)], axis=0)` . When the value of `shape` is `Tensor([])`, the shape of the return value is the same as that in PyTorch, which is the same as the shape of `rate`. The data type of the return value is determined by `dtype` .

There is no difference in function.

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | -            | shape         | The shape of the random number tensor sampled under each distribution under MindSpore, the shape of the return value is the same as PyTorch when the value `Tensor([])` |
|            | Parameter 2   | input        | rate          | Parameters of the Poisson distribution |
|            | Parameter 3   | generator    | seed          | MindSpore uses a random number seed to generate random numbers |
|            | Parameter 4   | -            | dtype         | The data type of the returned value in MindSpore supports int32/64, float16/32/64 |

## Code Example

```python
# PyTorch
import torch
import numpy as np

rate = torch.tensor(np.array([[5.0, 10.0], [5.0, 1.0]]), dtype=torch.float32)
output = torch.poisson(rate)
print(output.shape)
# torch.Size([2, 2])

# MindSpore
import mindspore as ms
import numpy as np

shape = ms.Tensor(np.array([]), ms.int32)
rate = ms.Tensor(np.array([[5.0, 10.0], [5.0, 1.0]]), dtype=ms.float32)
output = ms.ops.random_poisson(shape, rate, dtype=ms.float32)
print(output.shape)
# (2, 2)
```
