# 比较与torch.diag的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/MatrixDiag.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torch.diag

```python
torch.diag(
    input,
    diagonal=0,
    out=None
)
```

更多内容详见[torch.diag](https://pytorch.org/docs/1.5.0/torch.html#torch.diag)。

## mindspore.nn.MatrixDiag

```python
class mindspore.nn.MatrixDiag()(x)
```

更多内容详见[mindspore.nn.MatrixDiag](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/nn/mindspore.nn.MatrixDiag.html#mindspore.nn.MatrixDiag)。

## 使用方式

PyTorch: 仅支持1D和2D，如果输入是1D，则将返回一个2D的对角矩阵，除对角线外，均置0。如果输入是2D，则返回该矩阵的对角线上的值。同时，它支持通过参数`diagonal`指定对角线偏移量。

MindSpore：根据给定的值返回一个对角矩阵，对于k维的输入，将返回k+1维的对角矩阵。

## 代码示例

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