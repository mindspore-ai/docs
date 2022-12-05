# 比较与torch.cumprod的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/cumprod.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>
## torch.cumprod

```text
torch.cumprod(input, dim, *, dtype=None, out=None) -> Tensor
```

更多内容详见[torch.cumprod](https://pytorch.org/docs/1.8.1/generated/torch.cumprod.html)。

## mindspore.ops.cumprod

```text
mindspore.ops.cumprod(input, dim, dtype=None) -> Tensor
```

更多内容详见[mindspore.ops.cumprod](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.cumprod.html)。

## 差异对比

PyTorch：计算`input`沿着指定维度`dim`的元素累计积,`dtype`参数用于转换输入的数据格式。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数| 参数1 | input | input | - |
| | 参数2 | dim | dim | - |
| | 参数3 | dtype | dtype | - |
| | 参数4 | out | - | 不涉及 |

### 代码示例1

> PyTorch和MindSpore中此API实现相同功能。

```python
# PyTorch
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=int)
out = torch.cumprod(x, 0)
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
out = ops.cumprod(x, 0)
print(out)
# [[  1   2   3]
#  [  4  10  18]
#  [ 28  80 162]]
```

### 代码示例2

> PyTorch和MindSpore中此API参数`dtype`用于转换`input`的数据类型。

```python
# PyTorch
import torch
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=int)
out = torch.cumprod(x, 0, dtype=float)
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
out = ops.cumprod(x, 0, dtype=mindspore.float32)
print(out)
# [[  1.   2.   3.]
#  [  4.  10.  18.]
#  [ 28.  80. 162.]]
```
