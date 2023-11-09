# Differences with torch.Tensor.repeat

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_en/note/api_mapping/pytorch_diff/tensor_repeat.md)

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|   torch.Tensor.repeat    |   mindspore.Tensor.tile    |

## torch.Tensor.repeat

```text
torch.Tensor.repeat(*sizes) -> Tensor
```

For more information, see [torch.Tensor.repeat](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.repeat).

## mindspore.Tensor.tile

```text
mindspore.Tensor.tile(multiples)
```

For more information, see [mindspore.Tensor.tile](https://www.mindspore.cn/docs/en/r2.3/api_python/mindspore/Tensor/mindspore.Tensor.tile.html).

## Differences

The usage of `mindspore.Tensor.tile` is basically the same with that of `torch.Tensor.repeat`.

| Categories | Subcategories| PyTorch | MindSpore |Differences |
| ---- | ----- | ------- | --------- |------------------ |
| Parameters | Parameter 1 | *sizes   | multiples         | In PyTorch, parameter type is torch.Size or int and in MindSpore, parameter type must be tuple. |

### Code Example

```python
# PyTorch
import torch

input = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
output = input.repeat(16, 1)
print(output.shape)
# torch.Size([32, 2])

# MindSpore
import mindspore

x = mindspore.Tensor([[1, 2], [3, 4]], dtype=mindspore.float32)
output = x.tile((16, 1))
print(output.shape)
# (32, 2)
```
