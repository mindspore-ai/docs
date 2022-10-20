# 比较与torch.transpose的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Tensor.transpose.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source.png"></a>

## torch.transpose

```text
torch.transpose(input, dim0, dim1) -> Tensor
```

更多内容详见 [torch.transpose](https://pytorch.org/docs/1.8.1/generated/torch.transpose)。

## mindspore.ops.select

```text
mindspore.Tensor.transpose(*axes) -> Tensor
```

更多内容详见 [mindspore.Tensor.transpose](https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/Tensor/mindspore.Tensor.transpose.html)。

## 差异对比

PyTorch：对输入Tensor的指定两个维度之间进行转置。

MindSpore：MindSpore上不光可以在两个维度之间进行转置，还可以通过修改参数 *axes 在多个维度之间进行转置。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input |  |Tensor接口自己就是被操作对象，功能一致|
| | 参数2 | dim0 | - | PyTorch中，与dim1配合实现这两个维度之间的转置；MindSpore无此参数，可用axes实现同样的功能|
| | 参数3 | dim1 | - | PyTorch中，与dim0配合实现这两个维度之间的转置；MindSpore无此参数，可用axes实现同样的功能|
| | 参数4 | - | *axes | PyTorch无此参数，不过PyTorch中dim0和dim1可以实现此参数的部分功能|

### 代码示例1

说明：在使用torch.transpose(input, dim0, dim1)的时候，通过设置dim0和dim1实现input这两个维度之间的转置。MindSpore中虽然不能直接指定要进行转置的两个维度，但是可以通过调整axes参数实现同样的目的。假设input的shape为(3, 2, 1, 4)，dim0，dim1分别0，2，则会在第一和第三维之间进行转置，运算后的shape为(1, 2, 3, 4)；若要在MindSpore上实现这一操作，仅需要将axes设置为(2, 1, 0, 3)，即在默认维度(0, 1, 2, 3)基础上调换0和2的位置。
一般情况下，对于任意的n维input和有效的dim0，dim1，设置axes的时候，只需要在(0, ..., n-1)基础上将dim0, dim1对应的值交换位置即可。

```python
#PyTorch
import torch
import numpy as np

input = torch.tensor(np.arange(2*3*4).reshape(1, 2, 3, 4))
dim0 = 0
dim1 = 2
output =  torch.transpose(input, dim0, dim2)
print(output.numpy())
#[[[[ 0  1  2  3]]
#  [[12 13 14 15]]]
# [[[ 4  5  6  7]]
#  [[16 17 18 19]]]
# [[[ 8  9 10 11]]
#  [[20 21 22 23]]]]

#MindSpore
import mindspore as ms
from mindspore import Tensor
import numpy as np

input_x = Tensor(np.arange(2*3*4).reshape(1, 2, 3, 4))
axes = (2, 1, 0, 3)
output = input_x.transpose(axes)
print(output.asnumpy())
#[[[[ 0  1  2  3]]
#  [[12 13 14 15]]]
# [[[ 4  5  6  7]]
#  [[16 17 18 19]]]
# [[[ 8  9 10 11]]
#  [[20 21 22 23]]]]
```
