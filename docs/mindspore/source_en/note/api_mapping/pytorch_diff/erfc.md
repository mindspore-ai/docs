# Function Differences with torch.erfc

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/erfc.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.erfc

```text
torch.erfc(input, * , out=None) -> Tensor
```

For more information, see [torch.erfc](https://pytorch.org/docs/1.8.1/generated/torch.erfc.html).

## mindspore.ops.erfc

```text
mindspore.ops.erfc(x) -> Tensor
```

For more information, see [mindspore.ops.erfc](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.erfc.html).

## Differences

PyTorch: Compute the complementary error function for x element-wise, i.e. $ \operatorname{erfc}(x)=1-\frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^{2}} d t $ .

MindSpore: The function implemented with PyTorch is basically the same, but there are differences in the size of the supported dimensions.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | input | x |Same function, different parameter names, different size of supported dimensions |
| | Parameter 2 | out | - |Not involved |

### Code Example 1

> PyTorch does not limit the dimensionality of x, while MindSpore supports a dimensionality of x that must be less than 8. When the dimensionality of x is less than 8, the two APIs have same functions and have the same usage.

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

### Code Example 2

> When the dimension of x is more than or equal to 8, the same function can be achieved by API group sum. Use ops.reshape to reduce the dimension of x to 1, then call ops.erfc to compute it, and finally use ops.reshape again to up-dimension the result according to the original dimension of x.

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
