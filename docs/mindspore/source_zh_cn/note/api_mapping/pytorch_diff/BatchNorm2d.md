# 比较与torch.nn.BatchNorm2d的功能差异

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

更多内容详见 [torch.nn.BatchNorm2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.BatchNorm2d.html)。

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
)(input) -> Tensor
```

更多内容详见 [mindspore.nn.BatchNorm2d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.BatchNorm2d.html)。

## 差异对比

PyTorch：‎在四维输入(具有额外通道维度的小批量二维输入)上应用批归一化处理，以避免内部协变量偏移。

MindSpore：与PyTorch实现同样的功能。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | num_features | num_features | - |
| | 参数2 | input | input | - |
| | 参数3 | eps | eps | - |
| | 参数4 | momentum | momentum |功能相同，计算方式不同 |
| | 参数5 | affine | affine |- |
| | 参数6 | track_running_stats | use_batch_statistics | 功能相同，不同值对应的默认方式不同 |
| | 参数7 | - | gamma_init |γ 参数的初始化方法。默认值：’ones’。 |
| | 参数8 | - | beta_init |β 参数的初始化方法。默认值：’zeros’。 |
| | 参数9 | - | moving_mean_init |动态平均值的初始化方法。默认值：’zeros’。 |
| | 参数10 | - | moving_var_init |动态方差的初始化方法。默认值：’ones’。 |
| | 参数11 | - | data_format |数据格式可为’NHWC’或’NCHW’。默认值：’NCHW’。 |

### 代码示例1

> 功能相同，用法相同，PyTorch中的1-momentum等于MindSpore的momentum，都使用mini-batch数据和学习参数进行训练，这些参数见以下公式：

```text
    y= ( x−E[x] ) / ( √(Var[x]+ϵ) * γ + β )
```

```python
# PyTorch
from torch import nn,Tensor
import numpy as np

m = nn.BatchNorm2d(num_features=3,momentum=0.1)
inputx = Tensor(np.array([[[[0.1, 0.2],[0.3,0.4]],
                          [[0.5, 0.6],[0.7,0.8]],
                          [[0.9, 1],[1.1,1.2]]]]).astype(np.float32))
output = m(inputx)
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
from mindspore import Tensor,nn
import numpy as np

m = nn.BatchNorm2d(num_features=3,momentum=0.9)
m.set_train()
inputx = Tensor(np.array([[[[0.1, 0.2],[0.3,0.4]],
                          [[0.5, 0.6],[0.7,0.8]],
                          [[0.9, 1],[1.1,1.2]]]]).astype(np.float32))
output = m(inputx)
print(output)
# [[[[-1.3411045  -0.4470348 ]
#    [ 0.44703496  1.3411045 ]]
#
#   [[-1.341105   -0.4470351 ]
#    [ 0.44703424  1.3411041 ]]
#
#   [[-1.3411034  -0.44703388]
#    [ 0.44703573  1.3411053 ]]]
```
