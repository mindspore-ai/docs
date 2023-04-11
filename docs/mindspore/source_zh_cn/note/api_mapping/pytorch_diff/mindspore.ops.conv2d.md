# 比较与torch.nn.functional.conv2d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/mindspore.ops.conv2d.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.functional.conv2d

```text
torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```

更多内容详见[torch.nn.functional.conv2d](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.conv2d)。

## mindspore.ops.conv2d

```text
mindspore.ops.conv2d(inputs, weight, pad_mode="valid", padding=0, stride=1, dilation=1, group=1)
```

更多内容详见[mindspore.ops.conv2d](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.conv2d.html)。

## 差异对比

PyTorch：对输入Tensor计算二维卷积，需要手动传入卷积核的参数（即权重矩阵）。这个权重矩阵可以是一个预先训练好的模型参数，也可以是手动设定的矩阵。
MindSpore：与PyTorch实现的功能基本一致，但存在偏置差异，填充差异和参数名差异。

1. 偏置差异：MindSpore无bias参数
2. 填充差异：MindSpore默认对输入进行填充，而PyTorch则默认不填充。同时MindSpore填充模式可选项和行为与PyTorch不同，填充行为具体差异如下。

### 填充行为差异

 1. PyTorch的参数padding可选项有int，tuple of ints，默认为0，padding参数则用于控制填充的数量与位置。针对conv2d，padding指定为int的时候，会在输入的上下左右进行padding次的填充(若为默认值0则代表不进行填充)；padding指定为tuple的时候，会按照tuple的输入在上下和左右进行指定次数的填充；

 2. MindSpore的参数pad_mode可选项有‘same’，‘valid’，‘pad’，参数padding只能输入int或者tuple of ints，填充参数的详细意义如下：

     pad_mode设置为‘pad’的时候，MindSpore可以设置padding参数为大于等于0的正整数，会在输入的上下左右进行padding次的0填充(若为默认值0则代表不进行填充)，如果padding是一个由4个int组成的tuple，那么上、下、左、右的填充分别等于 padding[0] 、 padding[1] 、 padding[2] 和 padding[3]；pad_mode为另外两种模式时，padding参数必须只能设置为0，pad_mode设置为‘valid’模式时，不进行填充，只会在不超出特征图的范围内进行卷积；pad_mode设置为‘same’模式时，若需要padding的元素个数为偶数个，padding的元素则会均匀的分布在特征图的上下左右，若需要padding的元素个数为奇数个，MindSpore则会优先填充特征图的右侧和下侧(与PyTorch不同，类似TensorFlow)。

3. 参数名差异：PyTorch与MindSpore的参数名有一定差异，具体如下：

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | inputs |- |
| | 参数2 | weight | weight |- |
| | 参数3 | bias | - | MindSpore暂缺bias参数 |
| | 参数4 | stride | stride |- |
| | 参数5 | padding | padding |具体差异参考上文|
| | 参数6 | dilation | dilation |-|
| | 参数7 | groups | group |功能一致，参数名不同|
| | 参数9 | - | pad_mode |具体差异参考上文|

### 代码示例

> PyTorch的padding不为0时，需要MindSpore设置pad_mode为“pad”模式，若PyTorch将padding为(2, 3)，MindSpore需要将padding设置为(2, 2, 3, 3)。

```python
# PyTorch
import torch
import torch.nn.functional as F
import numpy as np

y = torch.tensor(np.ones([10, 32, 32, 32]), dtype=torch.float32)
weight = torch.tensor(np.ones([32, 32, 3, 3]), dtype=torch.float32)
output2 = F.conv2d(y, weight, padding=(2, 3))
print(output2.shape)
# torch.Size([10, 32, 34, 36])

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.ones([10, 32, 32, 32]), mindspore.float32)
weight = Tensor(np.ones([32, 32, 3, 3]), mindspore.float32)
output = ops.conv2d(x, weight, pad_mode="pad", padding=(2, 3))
print(output.shape)
# (10, 32, 34, 36)

```
