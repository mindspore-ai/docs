# 比较与torch.default_generator的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/default_generator.md)

## torch.default_generator

```text
class torch.default_generator
```

更多内容详见[torch.default_generator](https://pytorch.org/docs/1.8.1/torch.html#torch.torch.default_generator)。

## mindspore.nn.default_generator

```text
class mindspore.nn.default_generator
```

更多内容详见[mindspore.nn.default_generator](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/nn/mindspore.nn.default_generator.html)。

## 差异对比

PyTorch：`default_generator` 用于对默认生成器进行管理。当用户没有指定生成器时，随机算子会调用默认生成器来生成随机数。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。MindSpore返回 `mindspore.nn.Generator` 对象，PyTorch返回c++侧对象。

### 代码示例1

> 两API实现功能一致，用法相同。

```python
#PyTorch
import torch

torch_gen = torch.default_generator
print(type(torch_gen))
# <class 'torch._C.Generator'>


# MindSpore
from mindspore.nn import default_generator

ms_gen = default_generator()
print(type(ms_gen))
# <class 'mindspore.nn.generator.Generator'>
```
