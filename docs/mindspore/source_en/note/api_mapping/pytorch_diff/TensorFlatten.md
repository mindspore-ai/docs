# Function Differences with torch.Tensor.flatten

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/TensorFlatten.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.Tensor.flatten

```python
torch.Tensor.flatten(input, start_dim=0, end_dim=-1)
```

For more information, see [torch.Tensor.flatten](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.flatten).

## mindspore.Tensor.flatten

```python
mindspore.Tensor.flatten(order="C")
```

For more information, see [mindspore.Tensor.flatten](https://www.mindspore.cn/docs/en/master/api_python/mindspore/Tensor/mindspore.Tensor.flatten.html#mindspore.Tensor.flatten).

## Usage

`torch.flatten` restricts the range of dimensions that need to be extended by input parameters `start_dim` and `end_dim`.

`mindspore.Tensor.flatten` prioritizes row or column flatten by `order` to "C" or "F".

## Code Example

```python
import mindspore as ms

a = ms.Tensor([[1,2], [3,4]], ms.float32)
print(a.flatten())
# [1. 2. 3. 4.]
print(a.flatten('F'))
# [1. 3. 2. 4.]

import torch

b = torch.tensor([[[1, 2],[3, 4]],[[5, 6],[7, 8]]])
print(torch.Tensor.flatten(b))
# tensor([1, 2, 3, 4, 5, 6, 7, 8])
print(torch.Tensor.flatten(b, start_dim=1))
# tensor([[1, 2, 3, 4],
#         [5, 6, 7, 8]])
```

