# 比较与torch.nn.BatchNorm2d的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/BatchNorm2d.md)

## torch.nn.BatchNorm2d

```text
class torch.nn.BatchNorm2d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)(input) -> Tensor
```

更多内容详见[torch.nn.BatchNorm2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm2d.html)。

## mindspore.nn.BatchNorm2d

```text
class mindspore.nn.BatchNorm2d(
    num_features,
    eps=1e-5,
    momentum=0.9,
    affine=True,
    gamma_init='ones',
    beta_init='zeros',
    moving_mean_init='zeros',
    moving_var_init='ones',
    use_batch_statistics=None,
    data_format='NCHW'
)(x) -> Tensor
```

更多内容详见[mindspore.nn.BatchNorm2d](https://www.mindspore.cn/docs/zh-CN/r2.3/api_python/nn/mindspore.nn.BatchNorm2d.html)。

## 差异对比

PyTorch：在四维输入(具有额外mini-batch和channel通道的二维输入)上应用批归一化处理，以避免内部协变量偏移。

MindSpore：此API实现功能与PyTorch基本一致，典型区别有两点。MindSpore中momentum参数默认值为0.9，与PyTorch的momentum转换关系为1-momentum，默认值行为与PyTorch相同；训练以及推理时的参数更新策略和PyTorch有所不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | num_features | num_features | - |
| | 参数2 | eps | eps | - |
| | 参数3 | momentum | momentum | 功能一致，但PyTorch中的默认值是0.1，MindSpore中是0.9，与PyTorch的momentum转换关系为1-momentum，默认值行为与PyTorch相同 |
| | 参数4 | affine | affine |- |
| | 参数5 | track_running_stats | use_batch_statistics | 功能一致，不同值对应的默认方式不同 |
| | 参数6 | - | gamma_init |γ 参数的初始化方法，默认值："ones"。PyTorch无此参数 |
| | 参数7 | - | beta_init |β 参数的初始化方法，默认值："zeros" 。PyTorch无此参数 |
| | 参数8 | - | moving_mean_init |动态平均值的初始化方法，默认值："zeros"。PyTorch无此参数 |
| | 参数9 | - | moving_var_init |动态方差的初始化方法，默认值："ones"。PyTorch无此参数 |
| | 参数10 | - | data_format |MindSpore可指定输入数据格式可为"NHWC"或"NCHW"，默认值："NCHW"。PyTorch无此参数|
| 输入 | 单输入 | input | x | 功能一致，参数名不同 |

详细区别如下：
BatchNorm是CV领域比较特殊的正则化方法，它在训练和推理的过程中有着不同计算流程，通常由算子属性控制。MindSpore和PyTorch的 BatchNorm在这一点上使用了两种不同的参数组。

- 差异一

  `torch.nn.BatchNorm2d` 在不同参数下的状态

  |training|track_running_stats|状态|
  |----|----|--------------------------------------|
  |True|True|期望中训练的状态，running_mean 和 running_var 会跟踪整个训练过程中 batch 的统计特性，而每组输入数据用当前 batch 的 mean 和 var 统计特性做归一化，然后再更新 running_mean 和 running_var。|
  |True|False|每组输入数据会根据当前 batch 的统计特性做归一化，但不会有 running_mean 和 running_var 参数了。|
  |False|True|期望中推理的状态，BN 使用 running_mean 和 running_var 做归一化，并且不会对其进行更新。|
  |False|False|效果同第二点，只不过处于推理状态，不会学习 weight 和 bias 两个参数。一般不采用该状态。|

  `mindspore.nn.BatchNorm2d` 在不同参数下的状态

  |use_batch_statistics|状态|
  |----|--------------------------------------|
  |True|期望中训练的状态，moving_mean 和 moving_var 会跟踪整个训练过程中 batch 的统计特性，而每组输入数据用当前 batch 的 mean 和 var 统计特性做归一化，然后再更新 moving_mean 和 moving_var。
  |Fasle|期望中推理的状态，BN 使用 moving_mean 和 moving_var 做归一化，并且不会对其进行更新。
  |None|自动设置 use_batch_statistics。如果是训练，use_batch_statistics=True，如果是推理，use_batch_statistics=False。

  通过比较可以发现，`mindspore.nn.BatchNorm2d`  相比 `torch.nn.BatchNorm2d`，少了两种冗余的状态，仅保留了最常用的训练和推理两种状态。

- 差异二

  在PyTorch中，网络默认是训练模式，而MindSpore默认是推理模式（`is_training`为False），需要通过 `net.set_train()` 方法将网络调整为训练模式，此时才会在训练期间去对参数 `mean` 和 `variance` 进行计算，否则，在推理模式下，参数会尝试从checkpoint去加载。

- 差异三

  BatchNorm系列算子的momentum参数在MindSpore和PyTorch表示的意义相反，关系为：
  $$momentum_{pytorch} = 1 - momentum_{mindspore}$$

### 代码示例

> PyTorch中，1-momentum后的值等于MindSpore的momentum，都使用mini-batch数据和学习参数进行训练。

```python
# PyTorch
from torch import nn, tensor
import numpy as np

m = nn.BatchNorm2d(num_features=3, momentum=0.1)
input_py = tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_py)
print(output.detach().numpy())
# [[[[-1.3411044  -0.44703478]
#    [ 0.4470349   1.3411044 ]]
#
#   [[-1.3411043  -0.44703442]
#    [ 0.44703496  1.3411049 ]]
#
#   [[-1.3411039  -0.44703427]
#    [ 0.44703534  1.341105  ]]]]

# MindSpore
from mindspore import Tensor, nn
import numpy as np

m = nn.BatchNorm2d(num_features=3, momentum=0.9)
m.set_train()
#  BatchNorm2d<num_features=3, eps=1e-05, momentum=0.9, gamma=Parameter (name=gamma, shape=(3,), dtype=Float32, requires_grad=True), beta=Parameter (name=beta, shape=(3,), dtype=Float32, requires_grad=True), moving_mean=Parameter (name=mean, shape=(3,), dtype=Float32, requires_grad=False), moving_variance=Parameter (name=variance, shape=(3,), dtype=Float32, requires_grad=False)>

input_ms = Tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_ms)
print(output)
# [[[[-1.3411045  -0.4470348 ]
#    [ 0.44703496  1.3411045 ]]
#
#   [[-1.341105   -0.4470351 ]
#    [ 0.44703424  1.3411041 ]]
#
#   [[-1.3411034  -0.44703388]
#    [ 0.44703573  1.3411053 ]]]]
```
