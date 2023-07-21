# Differences with torch.range

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/range.md)

## torch.range

```python
torch.range(start=0,
            end,
            step=1,
            *,
            out=None,
            dtype=None,
            layout=torch.strided,
            device=None,
            requires_grad=False
            )
```

For more information, see [torch.range](https://pytorch.org/docs/1.8.1/generated/torch.range.html#torch.range).

## mindspore.ops.range

```python
mindspore.ops.range(start,
                    end,
                    step
                    )
```

For more information, see [mindspore.ops.range](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.range.html).

## Differences

API function of MindSpore is consistent with that of PyTorch.

MindSpore: The dtype of output Tensor is the same as input Tensor.

PyTorch: the dtype of output Tensor is determined by the parameter `dtype`.

| Categories | Subcategories | PyTorch       | MindSpore | Difference                                            |
|------------|---------------|---------------|-----------|-------------------------------------------------------|
| input      | input 1       | start         | start     | The data type of `start` parameter in MindSpore is Tensor and the `start` has no default value, while the data type of `start` parameter in PyTorch is float and the default value is 0 |
|            | input 2       | end           | end       | The data type of `end` parameter in MindSpore is Tensor, while the data type of `end` parameter in PyTorch is float |
|            | input 3       | step          | step      | The data type of `step` parameter in MindSpore is Tensor and the `step` has no default value, while the data type of `step` parameter in PyTorch is float and the default value is 0 |
|            | input 4       | out           | -         | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |
|            | input 5       | dtype         | -         | The dtype of output Tensor in MindSpore is the same as input Tensor，while the dtype of output Tensor in PyTorch is determined by the parameter `dtype` |
|            | input 6       | layout        | -         | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |
|            | input 7       | device        | -         | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |
|            | input 8       | requires_grad | -         | For details, see [General Difference Parameter Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html#general-difference-parameter-table) |

## Code Example

```python
# PyTorch
import torch

output = torch.range(0, 10, 4, dtype=torch.float32)
print(output)
# tensor([0., 4., 8.])

# MindSpore
import mindspore as ms
from mindspore import Tensor, ops

start = Tensor(0, ms.float32)
limit = Tensor(10, ms.float32)
delta = Tensor(4, ms.float32)
output = ops.range(start, limit, delta)
print(output)
# [0. 4. 8.]
```
