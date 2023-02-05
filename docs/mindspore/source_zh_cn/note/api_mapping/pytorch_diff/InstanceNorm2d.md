# 比较与torch.nn.InstanceNorm2d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/InstanceNorm2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.InstanceNorm2d

```text
class torch.nn.InstanceNorm2d(
    num_features,
    eps=1e-05,
    momentum=0.1,
    affine=False,
    track_running_stats=False
)(input) -> Tensor
```

更多内容详见[torch.nn.InstanceNorm2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.InstanceNorm2d.html)。

## mindspore.nn.InstanceNorm2d

```text
class mindspore.nn.InstanceNorm2d(
    num_features,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    gamma_init='ones',
    beta_init='zeros'
)(x) -> Tensor
```

更多内容详见[mindspore.nn.InstanceNorm2d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.InstanceNorm2d.html)。

## 差异对比

PyTorch：在四维输入(具有额外mini-batch和channel通道的二维输入)的每个channel内部应用归一化。

MindSpore：此API实现功能与PyTorch基本一致，典型区别有两点。MindSpore中affine参数默认值为True，会对内部参数 γ 和 β 进行学习，PyTorch默认值为False，不进行参数学习；PyTorch支持track_running_stats参数，如果设置为True，会在推理中使用训练得到的均值和方差，默认值为False，MindSpore无此参数，在训练和推理中都会使用输入数据的计算均值和方差，与PyTorch的默认值行为相同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 | input | x | 接口输入，功能一致，仅参数名不同 |
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

m = nn.InstanceNorm2d(num_features=3)
input_x = tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_x)
print(output.detach().numpy())
# [[[[-1.3411044  -0.44703478]
#    [ 0.4470349   1.3411044 ]]
#
#   [[-1.3411045  -0.44703463]
#    [ 0.44703472  1.3411046 ]]
#
#   [[-1.3411034  -0.44703388]
#    [ 0.44703573  1.3411053 ]]]]

# MindSpore
from mindspore import Tensor, nn
import numpy as np

m = nn.InstanceNorm2d(num_features=3, affine=False)
m.set_train()
input_x = Tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_x)
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
