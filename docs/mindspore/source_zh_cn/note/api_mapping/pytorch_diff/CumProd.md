# 比较与torch.cumprod的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/CumProd.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.cumprod

```text
torch.cumprod(input, dim, dtype=None) -> Tensor
```

更多内容详见 [torch.cumprod](https://pytorch.org/docs/1.8.1/generated/torch.cumprod.html)。

## mindspore.ops.CumProd

```text
class mindspore.ops.CumProd(exclusive=False, reverse=False)(x, axis) -> Tensor
```

更多内容详见 [mindspore.ops.CumProd](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.CumProd.html)。

## 差异对比

PyTorch：计算`input`沿着指定维度`dim`的元素累计积,`dtype`参数用于转换输入的数据格式。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，MindSpore不存在`dtype`参数；但同时增加了参数`exclusive`用于控制是否排除末尾元素计算元素累计积，以及参数`reverse`用于控制是否沿`axis`反转结果。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数| 参数1 | input | x |功能一致， 参数名不同 |
| | 参数2 | dim | axis | 功能一致，参数名不同|
| | 参数3 | dtype | - | PyTorch中此参数用于转换`input`的数据类型，MindSpore中无此参数|
| | 参数4 | - | exclusive | PyTorch中无此参数，MindSpore中的此参数控制是否排除末尾元素计算元素累计积|
| | 参数5 | - | reverse | PyTorch中无此参数，MindSpore中的此参数控制是否沿`axis`反转结果|

### 代码示例1

> MindSpore此API的参数`exclusive`和`reverse`均为默认值时，PyTorch和MindSpore中此API实现相同功能。

```python
# PyTorch
import torch
input = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=int)
out = torch.cumprod(input, dim=0)
out = out.detach().numpy()
print(out)
# [[  1   2   3]
#  [  4  10  18]
#  [ 28  80 162]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype('int32')
op = ops.CumProd()
output = ops(x, 0)
print(output)
# [[  1   2   3]
#  [  4  10  18]
#  [ 28  80 162]]
```

### 代码示例2

> PyTorch中此API参数`dtype`用于转换`input`的数据类型，MindSpore中此API无`dtype`参数，但是可以先转换输入参数的数据类型再进行输入，从而实现相同的功能。

```python
# PyTorch
import torch
input = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=int)
out = torch.cumprod(input, dim=0, dtype=float)
out = out.detach().numpy()
print(out)
# [[  1.   2.   3.]
#  [  4.  10.  18.]
#  [ 28.  80. 162.]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype('int32')
x = x.astype('int32')
op = ops.CumProd()
output = ops(x, 0)
print(output)
# [[  1.   2.   3.]
#  [  4.  10.  18.]
#  [ 28.  80. 162.]]
```
