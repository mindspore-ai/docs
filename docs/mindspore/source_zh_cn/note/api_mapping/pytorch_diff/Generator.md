# 比较与torch.Generator的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Generator.md)

## torch.Generator

```text
class torch.Generator(device='cpu')
```

更多内容详见[torch.Generator](https://pytorch.org/docs/1.8.1/generated/torch.Generator.html)。

## mindspore.nn.Generator

```text
class mindspore.nn.Generator()
```

更多内容详见[mindspore.nn.Generator](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/nn/mindspore.nn.Generator.html)。

## 差异对比

PyTorch：`Generator` 是管理随机状态的生成器，使用 `ByteTensor` 类型的Tensor保存生成器状态。生成器状态包含 `seed` 和 `offset` 两个值，其中 `seed` 为随机算子提供生成随机序列的种子，`offset` 提供随机序列上的偏移量。当 `seed` 和 `offset` 固定时，随机算子生成固定值。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，但用法上有差异。MindSpore管理随机状态的生成器在python侧实现，继承 `mindspore.nn.Cell` 。PyTorch在c++侧实现，python侧封装接口。PyTorch使用 `ByteTensor` 将 `seed` 和 `offset` 保存在一个Tensor中，MindSpore则将 `seed` ， `offset` 分开存储。PyTorch在c++侧更新 `offset` ，MindSpore重写 `mindspore.nn.Cell` 的 `construct` 方法更新 `offset` 。

| 分类 | 子类  | PyTorch | MindSpore   | 差异                                                         |
| ---- | ----- | ------- | ----------- | ------------------------------------------------------------ |
|   参数   | 参数1 |    device     | - | PyTorch设定生成器的后端，MindSpore无此参数 |
| 输入 | 单输入 | -      | step           | PyTorch在后端更新 `offset`，MindSpore对 `offset` 增加 `step` 值     |
| 方法 | - | device | - |  PyTorch返回生成器的后端，MindSpore无此方法  |
| 方法 | - | get_state | get_state |  PyTorch返回 `ByteTensor` 类型的值，MindSpore返回包含 `seed` 和 `offset` 值的Tensor的tuple |
| 方法 | - | initial_seed | initial_seed |  PyTorch返回int类型的 `seed` ，MindSpore返回Tensor类型的 `seed` |
| 方法 | - | manual_seed | manual_seed | 功能一致  |
| 方法 | - | seed | seed |  PyTorch通过c++后端随机生成int类型的 `seed` ，MindSpore 随机生成Tensor类型的 `seed` 。MindSpore通过numpy生成的随机种子，固定numpy的随机性可固定该接口的随机性 |
| 方法 | - | set_state | set_state |  PyTorch传入 `ByteTensor` 类型保存的状态，MindSpore传入 `seed` 和 `offset` 保存的状态 |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
#PyTorch
import torch

torch_gen = torch.Generator(device="cpu")
print(torch_gen.device)
# cpu
torch_gen.manual_seed(12)
print(torch_gen.initial_seed())
# 12
torch_state = torch_gen.get_state()
print(torch_state)
# tensor([12,  0,  0,  ...,  0,  0,  0], dtype=torch.uint8)
torch_gen.seed()
torch_gen.set_state(torch_state)
print(torch_gen.get_state())
# tensor([12,  0,  0,  ...,  0,  0,  0], dtype=torch.uint8)


# MindSpore
from mindspore.nn import Generator

ms_gen = Generator()
ms_gen.manual_seed(12)
print(ms_gen.initial_seed())
# 12
ms_seed, ms_offset = ms_gen.get_state()
print(ms_seed)
# 12
print(ms_offset)
# 0
ms_gen.seed()
ms_gen.set_state(ms_seed, ms_offset)
print(ms_gen.get_state())
#(Tensor(shape=[], dtype=Int32, value= 12), Tensor(shape=[], dtype=Int32, value= 0))
```
