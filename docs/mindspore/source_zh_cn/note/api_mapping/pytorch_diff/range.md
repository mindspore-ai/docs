# 比较与torch.range的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/range.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png"></a>

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

更多内容详见[torch.range](https://pytorch.org/docs/1.8.1/generated/torch.range.html#torch.range)。

## mindspore.ops.range

```python
mindspore.ops.range(start,
                    end,
                    step
                    )
```

更多内容详见[mindspore.ops.range](https://www.mindspore.cn/docs/zh-CN/r1.11/api_python/ops/mindspore.ops.range.html)。

## 差异对比

MindSpore: 输出Tensor的dtype取决于输入。

PyTorch: 输出Tensor的dtype取决于参数 `dtype` 。

| 分类  | 子类   | PyTorch       | MindSpore | 差异                                 |
|-----|------|---------------|-----------|------------------------------------|
| 输入  | 输入 1 | start         | start     | MindSpore必须为Tensor，然而PyTorch为float |
|     | 输入 2 | end           | end       | MindSpore必须为Tensor，然而PyTorch为float |
|     | 输入 3 | step          | step      | MindSpore必须为Tensor，然而PyTorch为float |
|     | 输入 4 | out           | -         | 不涉及                                |
|     | 输入 5 | dtype         | -         | 不涉及                                |
|     | 输入 6 | layout        | -         | 不涉及                                |
|     | 输入 7 | device        | -         | 不涉及                                |
|     | 输入 8 | requires_grad | -         | 不涉及                                |

## 代码示例

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
