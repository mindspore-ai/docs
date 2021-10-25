# Function Differences with torch.Tensor.stride

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/stride.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

## torch.Tensor.stride

```python
torch.Tensor.stride()
```

## mindspore.Tensor.strides

```python
mindspore.Tensor.strides()
```

## Differences

PyTorch: The number of elements that need to be traversed in each dimension, and the return type is a tuple.

MindSpore: The number of bytes that need to be traversed in each dimension, and the return type is a tuple.

## Code Example

```python
import mindspore as ms

a = ms.Tensor([[1, 2, 3], [7, 8, 9]])
print(a.strides)
# out:
# (24, 8)

import torch as tc

b = tc.tensor([[1, 2, 3], [7, 8, 9]])
print(b.stride())
# out:
# (3, 1)
```
