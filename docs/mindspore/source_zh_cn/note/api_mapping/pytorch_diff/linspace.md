# 比较与torch.linsapce的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/linspace.md)

## torch.linspace

```python
torch.linspace(start,
               end,
               steps,
               *,
               out=None,
               dtype=None,
               layout=torch.strided,
               device=None,
               requires_grad=False
              )
```

更多内容详见[torch.linspace](https://pytorch.org/docs/1.8.1/generated/torch.range.html#torch.linspace)。

## mindspore.ops.linspace

```python
mindspore.ops.linspace(start,
                       end,
                       steps
                      )
```

更多内容详见[mindspore.ops.linspace](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/ops/mindspore.ops.linspace.html)。

## 差异对比

MindSpore此API功能与PyTorch一致。

MindSpore: 输出Tensor的dtype与参数 `start` 相同。

PyTorch: 输出Tensor的dtype由参数 `dtype` 指定 。

| 分类  | 子类   | PyTorch       | MindSpore | 差异                                 |
|-----|------|---------------|-----------|------------------------------------|
| 输入  | 输入 1 | start         | start     | MindSpore中参数 `start` 的数据类型为Union[Tensor, int, float],PyTorch中参数 `start` 的数据类型为float |
|     | 输入 2 | end           | end       | MindSpore中参数 `end` 的数据类型为Union[Tensor, int, float]，PyTorch中参数 `end` 的数据类型为float |
|     | 输入 3 | steps         | steps     | MindSpore中参数 `steps` 的数据类型为Union[Tensor, int]，PyTorch中参数 `steps` 的数据类型为int |
|     | 输入 4 | out           | -         | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |
|     | 输入 5 | dtype         | -         | MindSpore输出Tensor的dtype与参数 `start`相同，PyTorch的输出Tensor的dtype由参数 `dtype`指定 |
|     | 输入 6 | layout        | -         | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |
|     | 输入 7 | device        | -         | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |
|     | 输入 8 | requires_grad | -         | 详见[通用差异参数表](https://www.mindspore.cn/docs/zh-CN/r2.3/note/api_mapping/pytorch_api_mapping.html#通用差异参数表) |

## 代码示例

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
