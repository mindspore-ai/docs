# Differences with torch.eq

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/expand.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.Tensor.expand

```text
torch.Tensor.expand(*sizes) → Tensor
```

For more information, see [torch.Tensor.expand](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.expand).

## mindspore.Tensor.expand

```text
mindspore.Tensor.expand(size) -> Tensor
```

For more information, see [mindspore.ops.expand](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.expand.html).

## Differences

API function of MindSpore is consistent with that of PyTorch.

PyTorch: The data types of parameter `sizes` are ``int`` or ``torch.Size`` .

MindSpore: The data type of parameter `size` is ``Tensor`` .

| Categories | Subcategories |PyTorch   | MindSpore | Difference |
| :-:       | :-:           | :-:       | :-:       |:-:        |
|Parameters | Parameter 1   | sizes     | size      | The data types supported by PyTorch are ``int`` or ``torch.Size`` , the data type supported by Mindspore is ``Tensor`` . |

### Code Example

```python
# PyTorch
import torch
import numpy as np
x = torch.tensor(np.array([[1], [2], [3]]), dtype=torch.float32)
size = (3, 4)
y = x.expand(size)
print(y)
# tensor([[1., 1., 1., 1.],
#         [2., 2., 2., 2.],
#         [3., 3., 3., 3.]])

# MindSpore
import mindspore as ms
import numpy as np
x = ms.Tensor(np.array([[1], [2], [3]]), dtype=ms.float32)
size = ms.Tensor(np.array([3, 4]), dtype=ms.int32)
y = x.expand(size)
print(y)
# [[1. 1. 1. 1.]
#  [2. 2. 2. 2.]
#  [3. 3. 3. 3.]]
```
