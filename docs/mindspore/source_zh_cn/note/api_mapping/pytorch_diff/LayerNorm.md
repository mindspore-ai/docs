# 比较与torch.nn.LayerNorm的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/LayerNorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.nn.LayerNorm

```python
class torch.nn.LayerNorm(
    normalized_shape,
    eps=1e-05,
    elementwise_affine=True
)(input) -> Tensor
```

更多内容详见[torch.nn.LayerNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.LayerNorm.html)。

## mindspore.nn.LayerNorm

```python
class mindspore.nn.LayerNorm(
    normalized_shape,
    begin_norm_axis=-1,
    begin_params_axis=-1,
    gamma_init='ones',
    beta_init='zeros',
    epsilon=1e-7
)(x) -> Tensor
```

更多内容详见[mindspore.nn.LayerNorm](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.LayerNorm.html)。

## 差异对比

PyTorch：在mini-batch输入上应用层归一化（Layer Normalization），其中参数`elementwise_affine`用于控制是否采用可学习参数。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，但MindSpore中不存在参数`elementwise_affine`，同时增加了参数`begin_norm_axis`控制归一化开始计算的轴，参数`begin_params_axis`控制第一个参数(beta, gamma)的维度，以及参数`gamma_init`和`beta_init`用来控制`γ`参数和`β`参数的初始化方法。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | normalized_shape | normalized_shape | - |
| | 参数2 | eps | epsilon | 功能一致，参数名不同，默认值不同 |
| | 参数3 | elementwise_affine | - | PyTorch中此参数用于控制是否采用可学习参数，MindSpore无此参数|
| | 参数4 | - | begin_norm_axis | MindSpore中的此参数控制归一化开始计算的轴，PyTorch无此参数|
| | 参数5 | - | begin_params_axis | MindSpore中的此参数控制第一个参数(beta, gamma)的维度，PyTorch无此参数|
| | 参数6 | - | gamma_init | MindSpore中的此参数控制`γ`参数的初始化方法，PyTorch无此参数|
| | 参数7 | - | beta_init | MindSpore中的此参数控制`β`参数的初始化方法，PyTorch无此参数|
|输入 | 单输入 | input | x | 功能一致，参数名不同|

### 代码示例

> PyTorch的参数elementwise_affine为True时，两API功能一致，用法相同。

```python
# PyTorch
import torch
import torch.nn as nn

inputs = torch.ones([20, 5, 10, 10])
m = nn.LayerNorm(inputs.size()[1:])
output = m(inputs)
print(output.detach().numpy().shape)
# (20, 5, 10, 10)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.numpy as np
import mindspore.nn as nn

x = Tensor(np.ones([20, 5, 10, 10]), mindspore.float32)
shape1 = x.shape[1:]
m = nn.LayerNorm(shape1, begin_norm_axis=1, begin_params_axis=1)
output = m(x).shape
print(output)
# (20, 5, 10, 10)
```
