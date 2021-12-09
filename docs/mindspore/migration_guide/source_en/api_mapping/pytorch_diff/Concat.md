# Function Differences with torch.cat

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/Concat.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## torch.cat

```python
torch.cat(
    tensors,
    dim=0,
    out=None
)
```

For more information, see [torch.cat](https://pytorch.org/docs/1.5.0/torch.html#torch.cat).

## mindspore.ops.Concat

```python
class mindspore.ops.Concat(
    axis=0
)(input_x)
```

For more information, see [mindspore.ops.Concat](https://mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.Concat.html#mindspore.ops.Concat).

## Differences

PyTorch: When the data type of the input tensors are different, the low-precision tensor will be automatically converted to a high-precision tensor.

MindSpore: Currently, the data type of the input tensors are required to remain the same. If not, the low-precision tensor can be converted to a high-precision tensor through ops.Cast and then call the Concat operator.

## Code Example

```python
import mindspore
import mindspore.ops as ops
from mindspore import Tensor
import torch
import numpy as np

# In MindSpore，converting low precision to high precision is needed before concat.
a = Tensor(np.ones([2, 3]).astype(np.float16))
b = Tensor(np.ones([2, 3]).astype(np.float32))
concat_op = ops.Concat()
cast_op = ops.Cast()
output = concat_op((cast_op(a, mindspore.float32), b))
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