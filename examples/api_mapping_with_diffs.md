# 比较与torch.where的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/examples/api_mapping_with_diffs.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.where

```text
torch.where(condition, x, y) -> Tensor
```

更多内容详见 [torch.where](https://pytorch.org/docs/1.8.1/generated/torch.where.html#torch.where)。

## mindspore.ops.select

```text
mindspore.ops.select(cond, x, y) -> Tensor
```

更多内容详见 [mindspore.ops.select](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.select.html)。

## 差异对比

PyTorch：根据条件判断Tensor中的元素的值来决定输出中的相应元素是从 x （如果元素值为True）还是从 y （如果元素值为False）中选择。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，不过广播机制有所差异。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | condition | cond |功能一致，参数名不同 |
| | 参数2 | x | x | PyTorch与MindSpore广播功能有差异|
| | 参数3 | y | y | PyTorch与MindSpore广播功能有差异|

### 代码示例1

说明：
PyTorch的参数x， y， condition三者shape不同的时候也支持广播。MindSpore目前支持x或者y为scalar，cond为tensor情况下的广播， 暂时不支持三者为shape不同的Tensor进行广播。
但是可以通过API组和实现同样的功能。当x或者y其中一个为Tensor且与cond形状不相同且可以广播时， 可以先将x或者y的shape使用ops.broadcast_to广播为cond的shape， 然后正常调用ops.select进行计算。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x_ = np.array([[np.arange(1, 13).reshape(3, 4),np.arange(-13, -1).reshape(3, 4)]])
y_ = np.ones((3, 4))
condition_ = np.array([[[[True, True, False, False], [True, False, False, False], [False, False, False, False]],
                           [[True, True, False, False], [True, True, False, False], [True, True, True, False]]]])
x = tensor(x_, dtype=torch.float32)
y = tensor(y_, dtype=torch.float32)
condition = tensor(condition_)
out = torch.where(condition, x, y).numpy()
print(out)
#[[[[  1.   2.   1.   1.]
#   [  5.   1.   1.   1.]
#   [  1.   1.   1.   1.]]
#
#  [[-13. -12.   1.   1.]
#   [ -9.  -8.   1.   1.]
#   [ -5.  -4.  -3.   1.]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x_ = np.array([[np.arange(1, 13).reshape(3, 4),np.arange(-13, -1).reshape(3, 4)]])
y_ = np.ones((3, 4))
condition_ = np.array([[[[True, True, False, False], [True, False, False, False], [False, False, False, False]],
                           [[True, True, False, False], [True, True, False, False], [True, True, True, False]]]])
x = Tensor(x_, mindspore.float32)
y = Tensor(y_, mindspore.float32)
cond = Tensor(condition_)
y_broadcasted = ops.broadcast_to(y, cond.shape)
output = ops.select(cond, x, y_broadcasted)
print(output)
#[[[[  1.   2.   1.   1.]
#   [  5.   1.   1.   1.]
#   [  1.   1.   1.   1.]]
#
#  [[-13. -12.   1.   1.]
#   [ -9.  -8.   1.   1.]
#   [ -5.  -4.  -3.   1.]]]]
```

### 代码示例2

说明：当输入y为Scalar的时候，两API实现相同的功能。

```python
# PyTorch
import torch
from torch import tensor

condition = tensor([True, False])
x = tensor([2, 3], dtype=torch.float32)
y = 2.0
out = torch.where(condition, x, y).numpy()
print(out)
# [2. 2.]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

cond = Tensor([True, False])
x = Tensor([2,3], mindspore.float32)
y = 2.0
output = ops.select(cond, x, y)
print(output)
# [2. 2.]
```

### 代码示例3

说明：当输入condition为表达式时，两API实现相同的功能。

```python
# PyTorch
import torch
from torch import tensor

x = tensor([1, 2, 3], dtype=torch.int32)
y = tensor([2, 4, 6], dtype=torch.int32)
out = torch.where(x + y <= 6, x, y).numpy()
print(out)
# [1 2 6]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops

x = Tensor([1, 2, 3], dtype=mindspore.int32)
y = Tensor([2, 4, 6], dtype=mindspore.int32)
output = ops.select(x + y <= 6, x, y)
print(output)
# [1 2 6]
```
