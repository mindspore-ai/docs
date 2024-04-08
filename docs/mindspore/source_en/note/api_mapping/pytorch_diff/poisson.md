# Differences with torch.poisson

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.q1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3.q1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/poisson.md)

## torch.poisson

```python
torch.poisson(input, generator=None)
```

For more information, see [torch.poisson](https://pytorch.org/docs/1.8.1/generated/torch.poisson.html).

## mindspore.ops.random_poisson

```python
mindspore.ops.random_poisson(shape, rate, seed=None, dtype=mstype.float32)
```

For more information, see [mindspore.ops.random_poisson](https://www.mindspore.cn/docs/en/r2.3.0rc1/api_python/ops/mindspore.ops.random_poisson.html).

## Differences

API function of MindSpore is consistent with that of PyTorch.

PyTorch: The shape and data type of the return value are the same as `input`.

MindSpore: `shape` determines the shape of the random number tensor sampled under each distribution, and the shape of the return value is `mindspore.concat([shape, mindspore.shape(rate)], axis=0)` . The data type of the return value is determined by `dtype` .

| Categories | Subcategories | PyTorch      | MindSpore     | Differences   |
| ---------- | ------------- | ------------ | ---------     | ------------- |
| Parameters | Parameter 1   | -            | shape         | The shape of the random number tensor sampled under each distribution under MindSpore |
|            | Parameter 2   | input        | rate          | Parameters of the Poisson distribution |
|            | Parameter 3   | generator    | seed          | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/r2.3.0rc1/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |
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

shape = ms.Tensor(np.array([1]), ms.int32)
rate = ms.Tensor(np.array([[5.0, 10.0], [5.0, 1.0]]), dtype=ms.float32)
output = ms.ops.random_poisson(shape, rate, dtype=ms.float32)
output = ms.ops.reshape(output, (2, 2))
print(output.shape)
# (2, 2)
```
