# 比较与torch.div的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/div.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.div

```text
torch.div(input, other, *, rounding_mode=None, out=None) -> Tensor
```

更多内容详见[torch.div](https://pytorch.org/docs/1.8.1/generated/torch.div.html)。

## mindspore.ops.div

```text
mindspore.ops.div(input, other, rounding_mode=None) -> Tensor
```

更多内容详见[mindspore.ops.div](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.div.html)。

## 差异对比

PyTorch：计算第一个输入除以第二个输入得到的商，其中商的取值方式取决于参数rounding_mode。

MindSpore：MindSpore的此API实现的功能与PyTorch一致。

| 分类 | 子类  | PyTorch       | MindSpore | 差异                                |
|:----|-----|:--------------|-----------|-----------------------------------|
| 参数| 参数1 | input         | input         | -                                 |
| | 参数2 | other         | other        | -                                 |
| | 参数3 | rounding_mode | rounding_mode | -                                 |
| | 参数4 | out           | -         |不涉及 |

### 代码示例1

> 当两个API的参数rounding_mode均为trunc时，两API均将除法得到的结果舍入到零。

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

x = tensor(np.array([1, -3, 8, 9]), dtype=torch.float32)
y = tensor(np.array([3, -2, -7, 5]), dtype=torch.float32)
out = torch.div(x, y, rounding_mode='trunc').detach().numpy()
print(out)
# [ 0.  1. -1.  1.]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x = Tensor(np.array([1, -3, 8, 9]), mindspore.float32)
y = Tensor(np.array([3, -2, -7, 5]), mindspore.float32)
output = ops.div(x, y, rounding_mode='trunc')
print(output)
# [ 0.  1. -1.  1.]
```

### 代码示例2

> 当两个API的参数rounding_mode均为floor时，两API均将除法得到的结果向下舍入。

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
output = ops.div(x, y, rounding_mode='floor')
print(output)
# [ 0.  1. -2.  1.]
```

### 代码示例3

> 当两个API的参数rounding_mode均为默认值None时，两API不对除法得到的结果做任何舍入操作。

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
#   [[-7. -6. -5.]
#    [-4. -3. -2.]]]]
```
