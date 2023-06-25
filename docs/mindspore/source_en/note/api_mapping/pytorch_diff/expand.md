# Differences with torch.Tensor.expand

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/expand.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.Tensor.expand

```text
torch.Tensor.expand(*sizes) -> Tensor
```

For more information, see [torch.Tensor.expand](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.expand).

## mindspore.Tensor.broadcast_to

```text
mindspore.Tensor.broadcast_to(shape) -> Tensor
```

For more information, see [mindspore.Tensor.broadcast_to](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.broadcast_to.html).

## Differences

MindSpore API function is consistent with PyTorch, with differences in the data types supported by the parameters.

PyTorch: `sizes` is the target shape after broadcasting, which can be of type ``torch.Size`` or a sequence consisting of ``int``.

MindSpore: `shape` is the target shape after broadcasting, which can be of type ``tuple[int]``.

| Categories | Subcategories | PyTorch | MindSpore | Differences  |
| --- |---------------|---------| --- |-------------|
| Parameter | Parameter 1 | *sizes | shape | Both parameters have different names, but both indicate the target shape after broadcasting. The type of `sizes` can be ``torch.Size`` or a sequence consisting of ``int``, and the type of `shape` can be ``tuple[int]``.|

### Code Example

```python
# PyTorch
import torch

x = torch.tensor([1, 2, 3])
output = x.expand(3, 3)
print(output)
print(value)
# tensor([[1, 2, 3],
#         [1, 2, 3],
#         [1, 2, 3]])

# MindSpore
import mindspore
from mindspore import Tensor

shape = (3, 3)
x = Tensor(np.array([1, 2, 3]))
output = x.broadcast_to(shape)
print(output)
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]]
```
