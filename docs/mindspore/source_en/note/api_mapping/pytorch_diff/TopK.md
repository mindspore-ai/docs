# Function Differences with torch.topk

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TopK.md)

## torch.topk

```python
torch.topk(
    input,
    k,
    dim=None,
    largest=True,
    sorted=True,
    out=None
)
```

For more information, see [torch.topk](https://pytorch.org/docs/1.5.0/torch.html#torch.topk).

## mindspore.ops.TopK

```python
class mindspore.ops.TopK(
    sorted=False
)(input_x, k)
```

For more information, see [mindspore.ops.TopK](https://mindspore.cn/docs/en/r1.9/api_python/ops/mindspore.ops.TopK.html#mindspore.ops.TopK).

## Differences

PyTorch: Support to obtain the maximum or minimum value of the first k entries of a specified dimension.

MindSpore：Currently, only the maximum value of the first k entries of the last dimension is supported.

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch

# In MindSpore, obtain the first k largest entries of the last dimension.
topk = ops.TopK()
k = 3
input_x = ms.Tensor([[1, 2, 3, 4], [2, 4, 6, 8]], ms.float16)
values, indices = topk(input_x, k)
print(values)
print(indices)

# Out：
# [[4. 3. 2.]]
# [[8. 6. 4.]]
# [[3 2 1]]
# [[3 2 1]]

# In torch, obtain the first k largest or smallest entries of a specific dimension.
# largest=True
input_x = torch.tensor([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=torch.float)
dim = 1
output = torch.topk(input_x, k, dim=dim, largest=True)
print(output)

# Out：
# torch.return_types.topk(
# values=tensor([[4., 3., 2.],
#                [8., 6., 4.]]),
# indices=tensor([[3, 2, 1],
#                 [3, 2, 1]]))

# largest=False
output = torch.topk(input_x, k, dim=dim, largest=False)
print(output)

# Out：
# torch.return_types.topk(
# values=tensor([[1., 2., 3.],
#                [2., 4., 6.]]),
# indices=tensor([[0, 1, 2],
#                 [0, 1, 2]]))
```
