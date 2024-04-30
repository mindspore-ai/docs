# 比较与torch.get_rng_state的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/get_rng_state.md)

## torch.get_rng_state

```text
class torch.get_rng_state
```

更多内容详见[torch.get_rng_state](https://pytorch.org/docs/1.8.1/generated/torch.get_rng_state.html)。

## mindspore.nn.get_rng_state

```text
class mindspore.nn.get_rng_state
```

更多内容详见[mindspore.nn.get_rng_state](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.get_rng_state.html)。

## 差异对比

PyTorch：`get_rng_state` 返回默认生成器的状态。生成器使用 `ByteTensor` 类型的Tensor保存状态。生成器状态包含 `seed` 和 `offset` 两个值，其中 `seed` 为随机算子提供生成随机序列的种子，`offset` 提供随机序列上的偏移量。当 `seed` 和 `offset` 固定时，随机算子生成固定值。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。MindSpore将 `seed` ， `offset` 分开存储，故MindSpore生成器状态返回包含 `seed` 和 `offset` 值的Tensor的tuple。

### 代码示例1

> 两API实现功能一致，用法相同。

```python
#PyTorch
import torch

torch.manual_seed(12)
torch_state = torch.get_rng_state()
print(torch_state)
# tensor([12,  0,  0,  ...,  0,  0,  0], dtype=torch.uint8)


# MindSpore
from mindspore.nn import manual_seed, get_rng_state

manual_seed(12)
ms_seed, ms_offset = get_rng_state()
print(ms_seed)
# 12
print(ms_offset)
# 0
```
