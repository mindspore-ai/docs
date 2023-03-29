# Function Differences with torch.range

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/range.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

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

MindSpore: The dtype of the output tensor depends on the input tensor.

PyTorch: The dtype of the output tensor depends on the parameter `dtype`.

| Categories | Subcategories | PyTorch       | MindSpore | Difference                                            |
|------------|---------------|---------------|-----------|-------------------------------------------------------|
| input      | input 1       | start         | start     | MindSpore must be a Tensor, whereas, PyTorch is float |
|            | input 2       | end           | end       | MindSpore must be a Tensor, whereas, PyTorch is float |
|            | input 3       | step          | step      | MindSpore must be a Tensor, whereas, PyTorch is float |
|            | input 4       | out           | -         | Not involved                                          |
|            | input 5       | dtype         | -         | Not involved                                          |
|            | input 6       | layout        | -         | Not involved                                          |
|            | input 7       | device        | -         | Not involved                                          |
|            | input 8       | requires_grad | -         | Not involved                                          |

## Code Example

```python
import mindspore as ms
import torch
from mindspore import Tensor, ops

# PyTorch
torch.range(0, 10, 4)
# tensor([0., 4., 8.])

# MindSpore
start = Tensor(0, ms.int32)
limit = Tensor(10, ms.int32)
delta = Tensor(4, ms.int32)
output = ops.range(start, limit, delta)
print(output)
# [0 4 8]
```
