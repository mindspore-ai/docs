# 比较与torch.nn.BatchNorm2d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/BatchNorm2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[mindspore.nn.BatchNorm2d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.BatchNorm2d.html)。

## 差异对比

PyTorch：在四维输入(具有额外通道维度的小批量二维输入)上应用批归一化处理，以避免内部协变量偏移。

MindSpore：此API实现功能与PyTorch基本一致，典型区别有两点。MindSpore中momentum参数默认值为0.9，与PyTorch的momentum转换关系为1-momentum，默认值行为与PyTorch相同；训练以及推理时的参数更新策略和PyTorch有所不同，详细区别请参考[与PyTorch典型区别-BatchNorm](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/typical_api_comparision.html#nn.BatchNorm2d)。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 | input | x | 接口输入，功能一致，仅参数名不同 |
| 参数 | 参数1 | num_features | num_features | - |
| | 参数2 | eps | eps | - |
| | 参数3 | momentum | momentum | 功能一致，但PyTorch中的默认值是0.1，MindSpore中是0.9，与PyTorch的momentum转换关系为1-momentum，默认值行为与PyTorch相同 |
| | 参数4 | affine | affine |- |
| | 参数5 | track_running_stats | use_batch_statistics | 功能一致，不同值对应的默认方式不同，详细区别请参考[与PyTorch典型区别-nn.BatchNorm2d](https://www.mindspore.cn/docs/zh-CN/master/migration_guide/typical_api_comparision.html#nn.BatchNorm2d)  |
| | 参数6 | - | gamma_init |γ 参数的初始化方法，默认值："ones" |
| | 参数7 | - | beta_init |β 参数的初始化方法，默认值："zeros" |
| | 参数8 | - | moving_mean_init |动态平均值的初始化方法，默认值："zeros" |
| | 参数9 | - | moving_var_init |动态方差的初始化方法，默认值："ones" |
| | 参数10 | - | data_format |MindSpore可指定输入数据格式可为"NHWC"或"NCHW"，默认值："NCHW"。PyTorch无此参数|

### 代码示例

> PyTorch中，1-momentum后的值等于MindSpore的momentum，都使用mini-batch数据和学习参数进行训练。

```python
# PyTorch
from torch import nn, Tensor
import numpy as np

m = nn.BatchNorm2d(num_features=3, momentum=0.1)
input_x = Tensor(np.array([[[[0.1, 0.2], [0.3, 0.4]],
                          [[0.5, 0.6], [0.7, 0.8]],
                          [[0.9, 1], [1.1, 1.2]]]]).astype(np.float32))
output = m(input_x)
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
#    [ 0.44703573  1.3411053 ]]]
```
