# Function Differences with torch.sparse_coo_tensor

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/SparseTensor.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## torch.sparse_coo_tensor

```python
torch.sparse_coo_tensor(
  indices,
  values,
  size=None,
  dtype=None,
  device=None,
  requires_grad=False
)
```

For more information, see [torch.sparse_coo_tensor](https://pytorch.org/docs/1.5.0/torch.html#torch.sparse_coo_tensor).

## mindspore.SparseTensor

```python
class mindspore.SparseTensor(
  indices,
  values,
  dense_shape
)
```

For more information, see [mindspore.SparseTensor](https://mindspore.cn/docs/api/en/master/api_python/mindspore/mindspore.SparseTensor.html#mindspore.SparseTensor).

## Differences

PyTorch: Constructs a sparse tensors in `COO(rdinate)` format.

MindSpore：Constructs a sparse tensors. It can only be used in the `Cell` construct method. PyNative mode does not support sparse tensor.

## Code Example

```python
# In MindSpore：
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import SparseTensor
class Net(nn.Cell):
    def __init__(self, dense_shape):
        super(Net, self).__init__()
        self.dense_shape = dense_shape
    def construct(self, indices, values):
        x = SparseTensor(indices, values, self.dense_shape)
        return x.indices, x.values, x.dense_shape

indices = Tensor([[0, 1], [1, 2]])
values = Tensor([1, 2], dtype=ms.float32)
out = Net((3, 4))(indices, values)
print(out[0])
print(out[1])
print(out[2])
# Out:
# [[0 1]
#  [1 2]]
# [1. 2.]
# (Tensor(shape=[], dtype=Int64, value= 3), Tensor(shape=[], dtype=Int64, value= 4))

# In torch:
import torch
i = torch.tensor([[0, 1],
                  [1, 2]])
v = torch.tensor([1, 2], dtype=torch.float32)
out = torch.sparse_coo_tensor(i, v, [3, 4])
print(out)
# Out:
# tensor(indices=tensor([[0, 1],
#                       [1, 2]]),
#        values=tensor([1., 2.]),
#        size=(3, 4), nnz=2, layout=torch.sparse_coo)
```
