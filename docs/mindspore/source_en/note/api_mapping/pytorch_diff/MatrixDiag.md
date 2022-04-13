# Function Differences with torch.diag

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/MatrixDiag.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.diag

```python
torch.diag(
    input,
    diagonal=0,
    out=None
)
```

For more information, see [torch.diag](https://pytorch.org/docs/1.5.0/torch.html#torch.diag).

## mindspore.nn.MatrixDiag

```python
class mindspore.nn.MatrixDiag()(x)
```

For more information, see [mindspore.nn.MatrixDiag](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.MatrixDiag.html#mindspore.nn.MatrixDiag).

## Differences

PyTorch: Only 1D and 2D are supported. If the input is a 1D Tensor, a 2D diagonal matrix will be returned, and all elements in the returned matrix are set to 0 except the diagonals. If the input is a 2D Tensor, the value on the diagonal of the matrix will be returned. It also supports diagonal offsets specified by parameter `diagonal`.

MindSpore：Returns a diagonal matrix based on the given value, and k+1 dimensional diagonal matrix for k dimensional input.

## Code Example

```python
import mindspore
from mindspore import Tensor, nn
import torch
import numpy as np

x1 = np.random.randn(2)
x2 = np.random.randn(2, 3)
x3 = np.random.randn(2, 3, 4)

# In MindSpore, for the given k-dimension input, a k+1 dimension diagonal matrix will be returned.
matrix_diag = nn.MatrixDiag()
for n, x in enumerate([x1, x2, x3]):
    try:
        input_x = Tensor(x, mindspore.float32)
        output = matrix_diag(input_x)
        print('input shape: {}; output size: {}'.format(
            str(n + 1), str(output.shape)
        ))
    except Exception as e:
        print('ERROR: ' + str(e))
# Out:
# input shape: 1; output size: (2, 2)
# input shape: 2; output size: (2, 3, 3)
# input shape: 3; output size: (2, 3, 4, 4)

# In torch, output for 1-dimension and 2-dimension input will be returned based on different rules.
# If the dimension of the input is greater than 2, it will raise error.
for n, x in enumerate([x1, x2, x3]):
    try:
        input_x = torch.tensor(x)
        output = torch.diag(input_x)
        print('input shape: {}; output size: {}'.format(
            str(n + 1), str(output.shape)
        ))
    except Exception as e:
        print('ERROR: ' + str(e))
# Out：
# input shape: 1; output size: torch.Size([2, 2])
# input shape: 2; output size: torch.Size([2])
# ERROR: matrix or a vector expected
```