# 比较与torch.nn.LeakyReLU的功能差异

## torch.nn.LeakyReLU

```text
class torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)(input) -> Tensor
```

更多内容详见 [torch.nn.LeakyReLU](https://pytorch.org/docs/1.8.1/generated/torch.nn.LeakyReLU.html)。

## mindspore.nn.LeakyReLU

```text
class mindspore.nn.LeakyReLU(alpha=0.2)(x) -> Tensor
```

更多内容详见 [mindspore.nn.LeakyReLU](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.LeakyReLU.html)。

## 差异对比

PyTorch：实现Leaky ReLU激活函数的基本功能，其中参数`negative_slope`是用于控制激活函数的斜率，参数`inplace`用于控制是否选择就地执行激活函数的操作。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，其中参数`alpha`与Pytorch中的参数`negative_slope`功能一致，参数名不同，默认值不同；但MindSpore不存在`inplace`参数。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | negative_slope | alpha |功能一致，参数名不同 |
| | 参数2 | inplace | - | PyTorch中此参数用于控制是否选择就地执行激活函数的操作，MindSpore无此参数|
| | 参数3 | input | x | 功能一致，参数名不同|

### 代码示例1

> PyTorch的参数`negative_slope`和MindSpore的参数`alpha`功能一致，参数名不同，默认值不同，当两者的值一致时，获得相同的结果。

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.LeakyReLU(0.2)
input = torch.tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]],dtype=float)
output = m(input).to(torch.float32).detach().numpy()
print(output)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

x = Tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype('float32')
m = nn.LeakyReLU()
output = m(x)
print(output)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]
```

### 代码示例2

> PyTorch的参数`inplace`用于控制是否选择就地执行激活函数的操作，也就是直接在输入上进行激活函数操作，输入被改变。MindSpore无此参数，但可以将输出赋值给输入实现类似的功能。

```python
# PyTorch
import torch
import torch.nn as nn

m = nn.LeakyReLU(0.2,inplace=True)
input = torch.tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]], dtype=torch.float32)
output = m(input)
print(output.detach().numpy())
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn

x = Tensor([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype('float32')
m = nn.LeakyReLU()
x = m(x)
print(x)
# [[-0.2  4.  -1.6]
#  [ 2.  -1.   9. ]]
```
