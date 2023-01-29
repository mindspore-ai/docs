# 比较与torch.erfc的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/erfc.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.erfc

```text
torch.erfc(input, * , out=None) -> Tensor
```

更多内容详见[torch.erfc](https://pytorch.org/docs/1.8.1/generated/torch.erfc.html)。

## mindspore.ops.erfc

```text
mindspore.ops.erfc(x) -> Tensor
```

更多内容详见[mindspore.ops.erfc](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/ops/mindspore.ops.erfc.html)。

## 差异对比

PyTorch：逐元素计算 x 的互补误差函数，即 $ \operatorname{erfc}(x)=1-\frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^{2}} d t $ 。

MindSpore：与PyTorch实现的功能基本一致，但支持的维度大小有差异。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | input | x |功能一致，参数名不同， 支持的维度大小有差异 |
|参数 | 参数2 | out | - |不涉及 |

### 代码示例1

> PyTorch没有限制x的维度，而MindSpore中x支持的维度必须小于8。当x的维度小于8时，两API功能一致，用法相同。

```python
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1))
x = tensor(x_, dtype=torch.float32)
out = torch.erfc(x).numpy()
print(out)
# [[[[[[[0.1572992]]]]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1))
x = Tensor(x_, mindspore.float32)
out = ops.erfc(x)
print(out)
# [[[[[[[0.1572992]]]]]]]
```

### 代码示例2

> 当x的维度超过或等于8时，可以通过API组和实现同样的功能。使用ops.reshape将x的维度降为1，然后调用ops.erfc进行计算，最后再次使用ops.reshape对得到的结果按照x的原始维度进行升维操作。

```python
import torch
from torch import tensor
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1, 1))
x = tensor(x_, dtype=torch.float32)
out = torch.erfc(x).numpy()
print(out)
# [[[[[[[[0.1572992]]]]]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import numpy as np

x_ = np.ones((1, 1, 1, 1, 1, 1, 1, 1))
x = Tensor(x_, mindspore.float32)
x_reshaped = ops.reshape(x, (-1,))
out_temp = ops.erfc(x_reshaped)
out = ops.reshape(out_temp, x.shape)
print(out)
# [[[[[[[[0.1572992]]]]]]]]
```
