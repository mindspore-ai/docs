# Function Differences with torch.cat

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_en/note/api_mapping/pytorch_diff/cat.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.cat

```python
torch.cat(
    tensors,
    dim=0,
    out=None
)
```

For more information, see [torch.cat](https://pytorch.org/docs/1.5.0/torch.html#torch.cat).

## mindspore.ops.cat

```python
class mindspore.ops.cat(
    tensors,
    axis=0
)
```

For more information, see [mindspore.ops.cat](https://mindspore.cn/docs/en/r2.0.0-alpha/api_python/ops/mindspore.ops.cat.html#mindspore.ops.cat).

## Differences

PyTorch: When the data type of the input tensors are different, the low-precision tensor will be automatically converted to a high-precision tensor.

MindSpore: Currently, the data type of the input tensors are required to remain the same. If not, the low-precision tensor can be converted to a high-precision tensor through ops.Cast and then call the cat operator.

## Code Example

```python
import mindspore
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore，converting low precision to high precision is needed before cat.
a = ms.Tensor(np.ones([2, 3]).astype(np.float16))
b = ms.Tensor(np.ones([2, 3]).astype(np.float32))
cast_op = ops.Cast()
output = ops.cat((cast_op(a, ms.float32), b))
print(output.shape)
# Out：
# (4, 3)

# In Pytorch.
a = torch.tensor(np.ones([2, 3]).astype(np.float16))
b = torch.tensor(np.ones([2, 3]).astype(np.float32))
output = torch.cat((a, b))
print(output.size())
# Out：
# torch.Size([4, 3])
```