# 比较与torch.nn.InstanceNorm1d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/InstanceNorm1d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.InstanceNorm1d

```text
class torch.nn.InstanceNorm1d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=False,
    track_running_stats=False
)(input) -> Tensor
```

更多内容详见[torch.nn.InstanceNorm1d](https://pytorch.org/docs/1.8.1/generated/torch.nn.InstanceNorm1d.html)。

## mindspore.nn.InstanceNorm1d

```text
class mindspore.nn.InstanceNorm1d(
    num_features,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    gamma_init='ones',
    beta_init='zeros'
)(x) -> Tensor
```

更多内容详见[mindspore.nn.InstanceNorm1d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.InstanceNorm1d.html)。

## 差异对比

PyTorch：对输入的二维或三维数据(具有额外mini-batch和channel通道的一维或具有mini-batch通道的二维输入)的每个channel内部应用归一化。

MindSpore：此API实现功能与PyTorch基本一致，但目前只能对三维数据进行归一化，典型区别有两点。MindSpore中affine参数默认值为True，会对内部参数 γ 和 β 进行学习，PyTorch默认值为False，不进行参数学习；PyTorch支持track_running_stats参数，如果设置为True，会在推理中使用训练得到的均值和方差，默认值为False，MindSpore无此参数，在训练和推理中都会使用输入数据的计算均值和方差，与PyTorch的默认值行为相同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 | input | x | 接口输入，功能基本一致，但PyTorch里允许输入是二维或三维的，而MindSpore里的输入只能是二维的 |
| 参数 | 参数1 | num_features | num_features | - |
| | 参数2 | eps | eps | - |
| | 参数3 | momentum | momentum | - |
| | 参数4 | affine | affine | 默认值不同，MindSpore默认值为True，会对内部参数 γ 和 β 进行学习，PyTorch默认值为False，不进行参数学习 |
| | 参数5 | track_running_stats | - | 如果设置为True，PyTorch会在推理中使用训练得到的均值和方差，默认值为False，MindSpore无此参数，在训练和推理中均会使用输入数据的计算均值和方差，与PyTorch的默认值False行为相同 |
| | 参数6 | - | gamma_init | 用于学习的变换参数 γ 初始化，默认是'ones'，而PyTorch不能额外设置，只能是'ones'|
| | 参数7 | - | beta_init |用于学习的变换参数 β 初始化，默认是'zeros'，而PyTorch不能额外设置，只能是'zeros' |

### 代码示例

> MindSpore的affine设置为False时，与PyTorch的默认行为功能一致。

```python
# PyTorch
from torch import nn, tensor
import numpy as np

m = nn.InstanceNorm1d(num_features=2)
input_x = tensor(np.array([[[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]]).astype(np.float32))
output = m(input_x)
print(output.detach().numpy())
# [[[ 1.5064616e+00 -9.0387678e-01 -9.0387678e-01  3.0129281e-01]
#   [-5.3444624e-01 -1.0688924e+00  3.2054459e-08  1.6033382e+00]]]

# MindSpore
from mindspore import Tensor, nn
import numpy as np

m = nn.InstanceNorm1d(num_features=2, affine=False)
m.set_train()
input_x = Tensor(np.array([[[0.7, 0.5, 0.5, 0.6], [0.5, 0.4, 0.6, 0.9]]]).astype(np.float32))
output = m(input_x)
print(output)
# [[[ 1.5064610e+00 -9.0387726e-01 -9.0387726e-01  3.0129224e-01]
#   [-5.3444624e-01 -1.0688924e+00  3.2054459e-08  1.6033382e+00]]]
```
