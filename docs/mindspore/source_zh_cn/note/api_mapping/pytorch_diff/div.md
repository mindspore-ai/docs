# 比较与torch.div的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/div.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.div

```text
torch.div(input, other, *, rounding_mode=None, out=None) -> Tensor
```

更多内容详见 [torch.div](https://pytorch.org/docs/1.8.1/generated/torch.div.html)。

## mindspore.ops.div

```text
mindspore.ops.div(x, y) -> Tensor
```

更多内容详见 [mindspore.ops.div](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.div.html)。

## 差异对比

PyTorch：计算第一个输入除以第二个输入得到的商，其中商的取值方式取决于参数rounding_mode。

MindSpore：MindSpore的此API只实现了PyTorch的API功能的一部分，即当PyTorch的API的参数rounding_mode取默认值时俩API功能一致。

| 分类 | 子类  | PyTorch       | MindSpore | 差异                                         |
|:----|-----|:--------------|-----------|--------------------------------------------|
| 参数| 参数1 | input         | x         | 功能一致， 参数名不同                                |
| | 参数2 | other         | y         | 功能一致， 参数名不同                                |
| | 参数3 | rounding_mode | -         | PyTorch中为可选参数，作用为设置结果的取整方式，MindSpore中无此参数|
| | 参数4 | out           | -         | PyTorch中表示结果Tensor，MindSpore中无此参数         |

### 代码示例1

> 当PyTorch的该API参数rounding_mode为trunc时，该API得到的结果需要向0取整。而MindSpore的此API只实现了不取整时的功能，但当输入Tensor为int类型时，可以用函数mindspore.ops.truncate_div同时实现Tensor相除并且结果向0取整的功能。若输入Tensor为其他数据类型，需要在自行将原始API得到的结果进行向零取整操作，目前MindSpore中无此类API。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x = tensor(np.array([1, -3, 8, 9]), dtype=torch.int32)
y = tensor(np.array([3, -2, -7, 5]), dtype=torch.int32)
out = torch.div(x, y, rounding_mode='trunc').detach().numpy()
print(out)
# [ 0.  1. -1.  1.]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.array([1, -3, 8, 9]), mindspore.int32)
y = Tensor(np.array([3, -2, -7, 5]), mindspore.int32)
output = ops.truncate_div(x, y)
print(output)
# [ 0.  1. -1.  1.]
```

### 代码示例2

> 当PyTorch的该API参数rounding_mode为floor时，API得到的结果需要向下取整。而MindSpore的此API只实现了不取整时的功能，可以用函数mindspore.ops.floor_div同时实现Tensor相除并且结果向下取整的功能。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x = tensor(np.array([1, -3, 8, 9]), dtype=torch.float32)
y = tensor(np.array([3, -2, -7, 5]), dtype=torch.float32)
out = torch.div(x, y, rounding_mode='floor').detach().numpy()
print(out)
# [ 0.  1. -2.  1.]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.array([1, -3, 8, 9]), mindspore.float32)
y = Tensor(np.array([3, -2, -7, 5]), mindspore.float32)
output = ops.floor_div(x, y)
print(output)
# [ 0.  1. -2.  1.]
```

### 代码示例3

> 当PyTorch中参数rounding_mode为默认值None时，两API实现相同的功能。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x = tensor(np.array([[np.arange(1, 7).reshape(2, 3), np.arange(-7, -1).reshape(2, 3)]]), dtype=torch.float32)
y = tensor(np.ones((2, 3)), dtype=torch.float32)
out = torch.div(x, y).detach().numpy()
print(out)
# [[[[ 1.  2.  3.]
#    [ 4.  5.  6.]]
#
#   [[-7. -6. -5.]
#    [-4. -3. -2.]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.array([[np.arange(1, 7).reshape(2, 3),np.arange(-7, -1).reshape(2, 3)]]), mindspore.float32)
y = Tensor(np.ones((2, 3)), mindspore.float32)
output = ops.div(x, y)
print(output)
# [[[[ 1.  2.  3.]
#    [ 4.  5.  6.]]
#
#   [[-7. -6. -5.]
#    [-4. -3. -2.]]]]
```
