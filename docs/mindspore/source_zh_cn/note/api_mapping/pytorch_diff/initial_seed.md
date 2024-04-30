# 比较与torch.initial_seed的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/initial_seed.md)

## torch.initial_seed

```text
class torch.initial_seed
```

更多内容详见[torch.initial_seed](https://pytorch.org/docs/1.8.1/generated/torch.initial_seed.html)。

## mindspore.nn.initial_seed

```text
class mindspore.nn.initial_seed
```

更多内容详见[mindspore.nn.initial_seed](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/nn/mindspore.nn.initial_seed.html)。

## 差异对比

PyTorch：`initial_seed` 返回默认生成器的初始化种子。生成器状态包含 `seed` 和 `offset` 两个值，其中 `seed` 为随机算子提供生成随机序列的种子，`offset` 提供随机序列上的偏移量。当 `seed` 和 `offset` 固定时，随机算子生成固定值。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。MindSpore返回Tensor类型，PyTorch返回int类型。

### 代码示例1

> 两API实现功能一致，用法相同。

```python
#PyTorch
import torch

torch.manual_seed(12)
torch_seed = torch.initial_seed()
print(torch_seed)
# 12


# MindSpore
from mindspore.nn import manual_seed, initial_seed

manual_seed(12)
ms_seed = initial_seed()
print(ms_seed)
# 12
```
