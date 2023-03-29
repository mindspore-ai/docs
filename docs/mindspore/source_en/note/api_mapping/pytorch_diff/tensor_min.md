# Function Differences with torch.Tensor.min

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/tensor_min.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

The following mapping relationships can be found in this file.

|     PyTorch APIs      |      MindSpore APIs       |
| :-------------------: | :-----------------------: |
|    torch.min     |  mindspore.ops.min   |
|   torch.Tensor.min    |   mindspore.Tensor.min    |

## torch.Tensor.min

```python
torch.Tensor.min(dim=None,
                 keepdim=False
                 )
```

For more information, see [torch.Tensor.min](https://pytorch.org/docs/1.8.1/tensors.html#torch.Tensor.min).

## mindspore.Tensor.min

```python
mindspore.Tensor.min(axis=None,
                     keepdims=False,
                     initial=None,
                     where=True)
```

For more information, see [mindspore.Tensor.min](https://www.mindspore.cn/docs/en/r2.0/api_python/mindspore/Tensor/mindspore.Tensor.min.html).

## Differences

MindSpore is compatible with Numpy parameters `initial` and `where` based on PyTorch.

| Categories | Subcategories | PyTorch | MindSpore | Differences  |
| --- |---------------|---------| --- |-------------|
| Inputs  | Input 1 | dim     | axis      | Same function, different parameter names |
|     | Input 2 | keepdim | keepdims  | Same function, different parameter names |
|     | Input 3 | - | initial        | Not involved        |
|     | Input 4 |  - | where      | Not involved        |

### Code Example 1

The two APIs implement the same functionality, and MindSpore includes an extension of Numpy.

```python
# PyTorch
import torch
from torch import tensor

a = tensor([[0.6750, 1.0857, 1.7197]])
output = a.min()
# tensor(0.6750)

# MindSpore
import mindspore
from mindspore import Tensor

a = Tensor([[0.6750, 1.0857, 1.7197]])
output = a.min()
print(output)
# 0.675
```
