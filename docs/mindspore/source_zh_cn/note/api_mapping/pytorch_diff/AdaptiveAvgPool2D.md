# 比较与torch.nn.AdaptiveAvgPool2d的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/AdaptiveAvgPool2D.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.AdaptiveAvgPool2d

```text
torch.nn.AdaptiveAvgPool2d(output_size) -> Tensor
```

更多内容详见 [torch.nn.AdaptiveAvgPool2d](https://pytorch.org/docs/1.8.1/generated/torch.nn.AdaptiveAvgPool2d.html)。

## mindspore.nn.AdaptiveAvgPool2d

```text
class mindspore.nn.AdaptiveAvgPool2d(output_size)(x) -> Tensor
```

更多内容详见 [mindspore.nn.AdaptiveAvgPool2d](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.AdaptiveAvgPool2d.html)。

## 差异对比

PyTorch：对输入3维或4维的Tensor，使用2维的自适应平均池化操作，指定输出的尺寸为H x W，输出的特征数目等于输入的特征数目。output_size可以是int类型的H和W组成的元组(H, W)，或者代表相同H和W的一个int值，或者None则表示输出大小将与输入相同。输入和输出数据格式可以是"NCHW"和"CHW"，N表示批处理大小、C是通道数、H是特征高度和W是特征宽度。

MindSpore：MindSpore此API实现功能与PyTorch一致，参数名也相同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 单输入 | input | x | 都是输入3维或4维的Tensor |
| 参数 | 参数1 | output_size | output_size | 功能一致，参数名相同 |

### 代码示例1

> 两API实功能一致，用法相同。输入为3维Tensor，数据尺寸为(C, H, W)，output_size=(None, new_W)，则Pytorch和MindSpore的AdaptiveAvgPool2D输出一致，数据尺寸为(C, H, new_W)。

```python
# case 1: output_size = (None, 2)
# PyTorch
import torch

# torch_input.shape = (1, 3, 3)
torch_input = torch.tensor([[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]]], dtype=torch.float32)
output_size = (None, 2)
torch_adaptive_avg_pool_2d = torch.nn.AdaptiveAvgPool2d(output_size)
# torch_output = (1, 3, 2)
torch_output = torch_adaptive_avg_pool_2d(torch_input)
torch_out_np = torch_output.numpy()
print(torch_out_np)
# [[[1.5 2.5]
#   [4.5 5.5]
#   [7.5 8.5]]]

# MindSpore
import numpy as np
import mindspore
from mindspore import Tensor

# ms_input.shape = (1, 3, 3)
ms_input = Tensor(np.array([[[1.0, 2.0, 3.0],
                             [4.0, 5.0, 6.0],
                             [7.0, 8.0, 9.0]]]), mindspore.float32)
output_size = (None, 2)
ms_adaptive_avg_pool_2d = mindspore.nn.AdaptiveAvgPool2d(output_size)
# ms_output = (1, 3, 2)
ms_output = ms_adaptive_avg_pool_2d(ms_input)
ms_out_np = ms_output.asnumpy()
print(ms_out_np)
# [[[1.5 2.5]
#   [4.5 5.5]
#   [7.5 8.5]]]
```
