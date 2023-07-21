# Differences with torch.bucketize

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/bucketize.md)

## torch.bucketize

```python
torch.bucketize(input, boundaries, *, out_int32=False, right=False, out=None)
```

For more information, see [torch.bucketize](https://pytorch.org/docs/1.8.1/torch.html#torch.bucketize).

## mindspore.ops.bucketize

```python
class mindspore.ops.bucketize(input, boundaries, *, right=False)
```

For more information, see [mindspore.ops.bucketize](https://mindspore.cn/docs/en/r2.1/api_python/ops/mindspore.ops.bucketize.html#mindspore.ops.bucketize).

## Usage

MindSpore API functions is consistent with that of PyTorch, with differences in the data types supported by the parameters.

PyTorch: `input` supports the scalar and Tensor types. `boundaries` supports the Tensor type, and the data type of the returned index can be specified via `out_int32`.

MindSpore: `input` supports Tensor type. `boundaries` supports list type, no `out_int32` parameter.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1 | input   | input         | Consistent functiona, different supported data types                    |
|      | Parameter 2 | boundaries   | boundaries      | Consistent function, different supported data types |
|      | Parameter 3 | out_int32   | - | PyTorch `out_int32` specifies the type of index to return, while MindSpore does not have this parameter. |
|      | Parameter 4 | right   | right | Consistent |
|      | Parameter 5 | out   | -         | PyTorch `out` can obtain outputs, while MindSpore does not have this parameter.|

## Code Example

```python
import torch

boundaries = torch.tensor([1, 3, 5, 7, 9])
v = torch.tensor([[3, 6, 9], [3, 6, 9]])
out1 = torch.bucketize(v, boundaries)
out2 = torch.bucketize(v, boundaries, right=True)
print(out1)
# out:
# tensor([[1, 3, 4],
#        [1, 3, 4]])

print(out2)
# out:
# tensor([[2, 3, 5],
#        [2, 3, 5]])

from mindspore import Tensor, ops
boundaries = [1, 3, 5, 7, 9]
v = Tensor([[3, 6, 9], [3, 6, 9]])
out1 = ops.bucketize(v, boundaries)
out2 = ops.bucketize(v, boundaries, right=True)
print(out1)
# out:
# [[1 3 4]
#  [1 3 4]]

print(out2)
# out:
# [[2 3 5]
#  [2 3 5]]
```