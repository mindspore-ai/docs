# Function Differences with torch.cumprod

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cumprod.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.cumprod

```text
torch.cumprod(input, dim, *, dtype=None, out=None) -> Tensor
```

For more information, see [torch.cumprod](https://pytorch.org/docs/1.8.1/generated/torch.cumprod.html).

## mindspore.ops.cumprod

```text
mindspore.ops.cumprod(input, dim, dtype=None) -> Tensor
```

For more information, see [mindspore.ops.cumprod](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.cumprod.html).

## Differences

PyTorch: Calculates the cumulative product of the elements of `input` along the specified dimension `dim`. The `dtype` parameter is used to convert the input data format.

MindSpore: MindSpore implements the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
| Parameters| Parameter 1 | input | input | - |
| | Parameter 2 | dim | dim | - |
| | Parameter 3 | dtype | dtype | - |
| | Parameter 4 | out | - | Not involved |

### Code Example 1

> This API implements the same function in PyTorch and MindSpore.

```python
# PyTorch
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=int)
out = torch.cumprod(x, 0)
out = out.detach().numpy()
print(out)
# [[  1   2   3]
#  [  4  10  18]
#  [ 28  80 162]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype('int32')
out = ops.cumprod(x, 0)
print(out)
# [[  1   2   3]
#  [  4  10  18]
#  [ 28  80 162]]
```

### Code Example 2

> This API parameter `dtype` in PyTorch and MindSpore is used to convert the data type of `input`.

```python
# PyTorch
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=int)
out = torch.cumprod(x, 0, dtype=float)
out = out.detach().numpy()
print(out)
# [[  1.   2.   3.]
#  [  4.  10.  18.]
#  [ 28.  80. 162.]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype('int32')
out = ops.cumprod(x, 0, dtype=mindspore.float32)
print(out)
# [[  1.   2.   3.]
#  [  4.  10.  18.]
#  [ 28.  80. 162.]]
```