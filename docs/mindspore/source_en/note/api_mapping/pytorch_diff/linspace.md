# Differences with torch.linspace

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/linspace.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.linspace

```python
torch.linspace(start,
               end,
               step,
               *,
               out=None,
               dtype=None,
               layout=torch.strided,
               device=None,
               requires_grad=False
             )
```

For more information, see [torch.linspace](https://pytorch.org/docs/1.8.1/generated/torch.linspace.html#torch.linspace).

## mindspore.ops.linspace

```python
mindspore.ops.linspace(start,
                       end,
                       steps
                      )
```

For more information, see [mindspore.ops.linspace](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.linspace.html).

## Differences

API function of MindSpore is consistent with that of PyTorch.

MindSpore: The dtype of output Tensor is the same as the parameter `start`.

PyTorch: the dtype of output Tensor is determined by the parameter `dtype`.

| Categories | Subcategories | PyTorch       | MindSpore | Difference                                            |
|------------|---------------|---------------|-----------|-------------------------------------------------------|
| input      | input 1       | start         | start     | The data type of `start` parameter in MindSpore is Union[Tensor, int, float], while the data type of `start` parameter in PyTorch is float |
|            | input 2       | end           | end       | The data type of `end` parameter in MindSpore is Union[Tensor, int, float], while the data type of `end` parameter in PyTorch is float |
|            | input 3       | step          | step      | The data type of `step` parameter in MindSpore is Union[Tensor, int], while the data type of `step` parameter in PyTorch is int |
|            | input 4       | out           | -         | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |
|            | input 5       | dtype         | -         | The dtype of output Tensor in MindSpore is the same as the parameter `start`，while the dtype of output Tensor in PyTorch is determined by the parameter `dtype` |
|            | input 6       | layout        | -         | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |
|            | input 7       | device        | -         | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |
|            | input 8       | requires_grad | -         | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |

## Code Example

```python
# PyTorch
import torch

output = torch.linspace(1, 10, 5, dtype=torch.float32)
print(output)
# tensor([1.0000, 3.2500, 5.5000, 7.7500, 10.0000])

# MindSpore
import mindspore as ms
from mindspore import Tensor, ops

start = Tensor(1, ms.float32)
limit = Tensor(10, ms.float32)
delta = Tensor(5, ms.int32)
output = ops.linspace(start, limit, delta)
print(output)
# [1. 3.25 5.5 7.75 10.]
```
