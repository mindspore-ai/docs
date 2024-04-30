# 比较与torch.seed的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/seed.md)

## torch.seed

```text
class torch.seed
```

更多内容详见[torch.seed](https://pytorch.org/docs/1.8.1/generated/torch.seed.html)。

## mindspore.nn.seed

```text
class mindspore.nn.seed
```

更多内容详见[mindspore.nn.seed](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/nn/mindspore.nn.seed.html)。

## 差异对比

PyTorch：`seed` 方法返回默认生成器生成的种子。默认生成器会将初始化种子设为该方法生成的种子。生成器状态包含 `seed` 和 `offset` 两个值，其中 `seed` 为随机算子提供生成随机序列的种子，`offset` 提供随机序列上的偏移量。当 `seed` 和 `offset` 固定时，随机算子生成固定值。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。PyTorch通过c++后端随机生成int类型的 `seed` ，MindSpore 随机生成Tensor类型的 `seed` 。MindSpore通过numpy生成的随机种子，固定numpy的随机性可固定该接口的随机性。

### 代码示例1

> 两API实现功能一致，用法相同。

```python
#PyTorch
import torch

torch_seed = torch.seed()
print(torch_seed)
# 937999739266300


# MindSpore
import numpy as np
from mindspore.nn import manual_seed, seed

np.random.seed(20)
ms_seed = seed()
print(ms_seed)
# 1663920602
```
