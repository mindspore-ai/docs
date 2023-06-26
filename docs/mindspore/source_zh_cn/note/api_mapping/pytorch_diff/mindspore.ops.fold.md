# 比较与torch.nn.functional.fold的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/mindspore.ops.fold.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.functional.fold

```text
torch.nn.functional.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)
```

更多内容详见[torch.nn.functional.fold](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.fold)。

## mindspore.ops.fold

```text
mindspore.ops.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)
```

更多内容详见[mindspore.ops.fold](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.fold.html)。

## 差异对比

PyTorch：将提取出的滑动局部区域块还原成更大的输出Tensor。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | input | Pytorch：shape大小为 :math:`(N, C \times \prod(\text{kernel_size}), L)` ，MindSpore：shape大小为 :math:`(N, C, \prod(\text{kernel_size}), L)` |
| | 参数2 | output_size | output_size | Pytorch：整型或者元组类型，MindSpore：一维Tensor，包含两个元素，均为整数类型 |
| | 参数3 | kernel_size | kernel_size |- |
| | 参数4 | dilation | dilation |- |
| | 参数5 | padding | padding |- |
| | 参数6 | stride | stride |- |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
import numpy as np
x = np.random.randn(1, 3 * 2 * 2, 12)
input = torch.tensor(x, dtype=torch.float32)
output = torch.nn.functional.fold(input, output_size=(4, 5), kernel_size=(2, 2))
print(output.detach().shape)
# torch.Size([1, 3, 4, 5])

# MindSpore
import mindspore
import numpy as np
x = np.random.randn(1, 3, 4, 12)
input = mindspore.Tensor(x, mindspore.float32)
output_size = mindspore.Tensor((4, 5), mindspore.int32)
output = mindspore.ops.fold(input, output_size, kernel_size=(2, 2))
print(output)
# (1, 3, 4, 5)
```
