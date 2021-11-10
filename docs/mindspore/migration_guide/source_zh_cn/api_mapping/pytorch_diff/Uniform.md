# 比较与torch.nn.init.uniform_的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/Uniform.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## torch.nn.init.uniform_

```python
torch.nn.init.uniform_(
    tensor,
    a=0.0,
    b=1.0
)
```

更多内容详见[torch.nn.init.uniform_](https://pytorch.org/docs/1.5.0/nn.init.html#torch.nn.init.uniform_)。

## mindspore.common.initializer.Uniform

```python
class mindspore.common.initializer.Uniform(scale=0.07)(arr)
```

更多内容详见[mindspore.common.initializer.Uniform](https://mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.common.initializer.html#mindspore.common.initializer.Uniform)。

## 使用方式

PyTorch：通过入参`a`和`b`分别指定均匀分布的上下界，即U(-a, b)。

MindSpore：仅通过一个入参`scale`指定均匀分布的范围，即U(-scale, scale)，且是原地更新输入值。

## 代码示例

```python
import mindspore
import torch
import numpy as np

# In MindSpore, only one parameter is set to specify the scope of uniform distribution (-1, 1).
input_x = np.array([1, 1, 1]).astype(np.float32)
uniform = mindspore.common.initializer.Uniform(scale=1)
uniform(input_x)
print(input_x)
# Out：
# [-0.2333 0.6208 -0.1627]

# In torch, parameters are set separately to specify the lower and upper bound of uniform distribution.
input_x = torch.tensor(np.array([1, 1, 1]).astype(np.float32))
output = torch.nn.init.uniform_(tensor=input_x, a=-1, b=1)
print(output)
# Out：
# tensor([0.9936, 0.7676, -0.8275])
```