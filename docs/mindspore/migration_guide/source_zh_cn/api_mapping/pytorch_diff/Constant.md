# 比较与torch.nn.init.constant_的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Constant.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.init.constant_

```python
torch.nn.init.constant_(
    tensor,
    val
)
```

更多内容详见[torch.nn.init.constant_](https://pytorch.org/docs/1.5.0/nn.init.html#torch.nn.init.constant_)。

## mindspore.common.initializer.Constant

```python
class mindspore.common.initializer.Constant(value)(arr)
```

更多内容详见[mindspore.common.initializer.Constant](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Constant)。

## 使用方式

PyTorch：以常量`val`填充输入的tensor。

MindSpore：以`value`（整型或numpy数组）填充输入的numpy数组，且是原地更新输入值。

## 代码示例

```python
import mindspore
import torch
import numpy as np

# In MindSpore, fill a constant array with value(int or numpy array).
input_constant = np.array([1, 2, 3])
constant_init = mindspore.common.initializer.Constant(value=1)
out_constant = constant_init(input_constant)
print(input_constant)
# Out：
# [1 1 1]

# In torch, fill in the input tensor with constant val.
input_constant = np.array([1, 2, 3])
out_constant = torch.nn.init.constant_(
    tensor=torch.tensor(input_constant),
    val=1
)
print(out_constant)
# Out：
# tensor([1, 1, 1])
```