# 比较与torch.range的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/range.md)

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

更多内容详见[mindspore.ops.range](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/ops/mindspore.ops.range.html)。

## 差异对比

MindSpore此API功能与PyTorch一致。

MindSpore: 输出Tensor的dtype与输入Tensor相同。

PyTorch: 输出Tensor的dtype由参数 `dtype` 指定 。

| 分类  | 子类   | PyTorch       | MindSpore | 差异                                 |
|-----|------|---------------|-----------|------------------------------------|
| 输入  | 输入 1 | start         | start     | MindSpore中参数 `start` 的数据类型为Tensor，无默认值。PyTorch中参数 `start` 的数据类型为float，默认值为0 |
|     | 输入 2 | end           | end       | MindSpore中参数 `end` 的数据类型为Tensor，PyTorch中参数 `end` 的数据类型为float |
|     | 输入 3 | step          | step      | MindSpore中参数 `step` 的数据类型为Tensor，无默认值。PyTorch中参数 `step` 的数据类型为float，默认值为1 |
|     | 输入 4 | out           | -         | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |
|     | 输入 5 | dtype         | -         | MindSpore输出Tensor的dtype与输入Tensor相同，PyTorch的输出Tensor的dtype由参数 `dtype`指定 |
|     | 输入 6 | layout        | -         | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |
|     | 输入 7 | device        | -         | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |
|     | 输入 8 | requires_grad | -         | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

## 代码示例

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
