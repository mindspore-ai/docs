# 比较与torch.nn.LocalResponseNorm的功能差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/LRN.md)

## torch.nn.LocalResponseNorm

```text
class torch.nn.LocalResponseNorm(
    size,
    alpha=0.0001,
    beta=0.75,
    k=1.0
)(input) -> Tensor
```

更多内容详见[torch.nn.LocalResponseNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.LocalResponseNorm.html)。

## mindspore.ops.LRN

```text
class mindspore.ops.LRN(
    depth_radius=5,
    bias=1.0,
    alpha=1.0,
    beta=0.5,
    norm_region="ACROSS_CHANNELS"
)(x) -> Tensor
```

更多内容详见[mindspore.ops.LRN](https://www.mindspore.cn/docs/zh-CN/r1.11/api_python/ops/mindspore.ops.LRN.html)。

## 差异对比

PyTorch：进行局部响应归一化操作，它通过特定的方式对每个神经元的输入进行归一化，以提高深度神经网络的泛化能力。返回一个与input具有相同类型的Tensor。

MindSpore：MindSpore此API实现功能与PyTorch一致。MindSpore的 `depth_radius` 参数与PyTorch的 `size` 实现同样的功能，但存在一个二倍映射关系：size=2*depth_radius。目前mindspore.ops.LRN与tf.raw_ops.LRN能完全对标，两者能达到相同的精度；如果与torch.nn.LocalResponseNorm相比，会存在1e-3的精度差异。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | size       | depth_radius         | 归一化的相邻神经元数，映射关系为 size=2*depth_radius|
|  | 参数2 | k       | bias         | 功能一致，参数名不同      |
|  | 参数3 | alpha       | alpha         | - |
|  | 参数4 | beta       | beta         | - |
|  | 参数5 | -       | norm_region         | 指定归一化区域。PyTorch无此参数 |
| 输入| 单输入 | input | x        | 功能一致，参数名不同           |

### 代码示例1

> MindSpore中的 `depth_radius` 与PyTorch的 `size` 存在二倍映射关系，因此将 `depth_radius` 设置为 `size` 的一半就能实现同样的功能。

```python
# PyTorch
import torch
import numpy as np

input_x = torch.from_numpy(np.array([[[[2.4], [3.51]],[[1.3], [-4.4]]]], dtype=np.float32))
output = torch.nn.LocalResponseNorm(size=2, alpha=0.0001, beta=0.75, k=1.0)(input_x)
print(output.numpy())
#[[[[ 2.3994818]
#   [ 3.5083795]]
#  [[ 1.2996368]
#   [-4.39478  ]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops.operations as ops
import numpy as np

input_x = Tensor(np.array([[[[2.4], [3.51]],[[1.3], [-4.4]]]]), mindspore.float32)
lrn = ops.LRN(depth_radius=1, bias=1.0, alpha=0.0001, beta=0.75)
output = lrn(input_x)
print(output)
#[[[[ 2.39866  ]
#   [ 3.5016835]]
#  [[ 1.2992741]
#   [-4.3895745]]]]
```
