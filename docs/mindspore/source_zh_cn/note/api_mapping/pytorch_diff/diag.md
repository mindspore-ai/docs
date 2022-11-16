# 比较与torch.diag的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/diag.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.diag

```text
torch.diag(input, diagonal=0) -> Tensor
```

更多内容详见 [torch.diag](https://pytorch.org/docs/1.8.1/generated/torch.diag.html)。

## mindspore.ops.diag

```text
mindspore.ops.diag(input_x) -> Tensor
```

更多内容详见 [mindspore.ops.diag](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.diag.html)。

## 差异对比

PyTorch：若输入为一维张量，用输入的对角线值构成的一维张量来构造对角线张量；若输入为矩阵，则返回由输入的对角线元素组成的一维张量。

MindSpore：MindSpore此API，若输入为一维张量，则实现与PyTorch相同的功能；若输入为矩阵，则不能实现与PyTorch相同的功能，且没有`diagonal`参数控制要考虑的对角线的位置。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | input_x |功能一致， 参数名不同 |
| | 参数2 | diagonal | - | PyTorch中`diagonal`的取值用于控制要考虑的对角线的位置，MindSpore无此参数|

### 代码示例1

> PyTorch的此API参数`x`支持多维张量和一维张量，且存在`diagonal`参数用于控制要考虑的对角线的位置，而MindSpore此API仅支持一维张量，不存在`diagonal`参数；当输入参数x为一维张量且`diagonal`为0时,两API实现相同的功能。

```python
# PyTorch
import torch
x = torch.tensor([1,2,3,4],dtype=int)
out = torch.diag(x)
out = out.detach().numpy()
print(out)
# [[1 0 0 0]
#  [0 2 0 0]
#  [0 0 3 0]
#  [0 0 0 4]]

# MindSpore
from mindspore import Tensor
import mindspore.ops as ops
input_x = Tensor([1, 2, 3, 4]).astype('int32')
output = ops.diag(input_x)
print(output)
# [[1 0 0 0]
#  [0 2 0 0]
#  [0 0 3 0]
#  [0 0 0 4]]

```

### 代码示例2

> 当输入参数`x`为一维张量且`diagonal`不为0时,PyTorch的此API可控制要考虑的对角线的位置，而MindSpore的此API没有`diagonal`参数，可以将此API得到的输出使用mindspore.ops.pad进行处理，从而实现相同功能。

```python
# PyTorch
import torch
x = torch.tensor([1,2,3,4],dtype=int)
# diagonal大于0时的结果
out = torch.diag(x, diagonal=1)
out = out.detach().numpy()
print(out)
# [[0 1 0 0 0]
#  [0 0 2 0 0]
#  [0 0 0 3 0]
#  [0 0 0 0 4]
#  [0 0 0 0 0]]

# diagonal小于0时的结果
out = torch.diag(x, diagonal=-1)
out = out.detach().numpy()
print(out)
# [[0 0 0 0 0]
#  [1 0 0 0 0]
#  [0 2 0 0 0]
#  [0 0 3 0 0]
#  [0 0 0 4 0]]

# MindSpore
from mindspore import Tensor
import mindspore.ops as ops
input_x = Tensor([1, 2, 3, 4]).astype('int32')
output = ops.diag(input_x)
# MindSpore对应于diagonal大于0时的此API功能实现
padding = ((0, 1), (1, 0))
a = ops.pad(output, padding)
print(a)
# [[0 1 0 0 0]
#  [0 0 2 0 0]
#  [0 0 0 3 0]
#  [0 0 0 0 4]
#  [0 0 0 0 0]]

# MindSpore对应于diagonal大于0时的此API功能实现
padding = ((1, 0), (0, 1))
a = ops.pad(output, padding)
print(a)
# [[0 0 0 0 0]
#  [1 0 0 0 0]
#  [0 2 0 0 0]
#  [0 0 3 0 0]
#  [0 0 0 4 0]]
```

### 代码示例3

> PyTorch的此API输入为矩阵且使用`diagonal`时,MindSpore此API不支持此功能，使用mindspore.numpy.diag算子可实现此功能。

```python
# PyTorch
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=int)
# diagonal大于0时的结果
out = torch.diag(x, diagonal=1)
out = out.detach().numpy()
print(out)
# [2 6]

# diagonal为默认值0时的结果
out = torch.diag(x)
out = out.detach().numpy()
print(out)
# [1 5 9]

# diagonal小于0时的结果
out = torch.diag(x, diagonal=-1)
out = out.detach().numpy()
print(out)
# [4 8]

# MindSpore
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.numpy as np
input_x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype('int32')
#对应于diagonal大于0时的mindspore.numpy.diag的此功能实现
output = np.diag(input_x, k=1)
print(output)
# [2 6]

#对应于diagonal默认为0时的mindspore.numpy.diag的此功能实现
output = np.diag(input_x)
print(output)
# [1 5 9]

#对应于diagonal小于0时的mindspore.numpy.diag的此功能实现
output = np.diag(input_x, k=-1)
print(output)
# [4 8]
```
