# 比较与torch.bmm的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/BatchMatMul.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.bmm

```text
torch.bmm(input, mat2, deterministic=False) -> Tensor
```

更多内容详见 [torch.bmm](https://pytorch.org/docs/1.8.1/generated/torch.bmm.html)。

## mindspore.ops.BatchMatMul

```text
mindspore.ops.BatchMatMul(transpose_a=False, transpose_b=False)(x, y) -> Tensor
```

更多内容详见 [mindspore.ops.BatchMatMul](https://mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.BatchMatMul.html)。

## 差异对比

PyTorch：对input和mat2执行批量矩阵乘积，其中input和mat2必须是3-D的tensor。如果input是一个(b, n, m)的tensor，mat2是一个(b, n, p)的tensor，两者矩阵乘积的结果out为(b, n, p)。

MindSpore: MindSpore此API实现功能与PyTorch基本一致， 不过MindSpore支持3D以及更高维度的矩阵乘法计算，其中MindSpore的transpose_a若为True，会把输入相乘的第一个tensor的最后两维进行交换。

| 分类 | 子类  | PyTorch       | MindSpore   | 差异                                                         |
| ---- | ----- | ------------- | ----------- | ------------------------------------------------------------ |
| 参数 | 参数1 | input         | x           | 功能一致，参数名不同                                         |
|      | 参数2 | mat_2         | y           | 功能一致，参数名不同                                         |
|      | 参数3 | deterministic | -           | 此参数只适用于稀疏的稀疏密集的CUDA bmm，MindSpore无此参数    |
|      | 参数4 | -             | transpose_a | transpose_a若为True，会把输入相乘的第一个tensor的最后两维进行交换。 |
|      | 参数5 | -             | transpose_b | transpose_b若为True，会把输入相乘的第二个tensor的最后两维进行交换。 |

### 代码示例1

两API实功能一致， 用法相同。

```python
# PyTorch
import numpy as np
import torch
from torch import tensor

input = torch.tensor(np.ones(shape=[2, 1, 4]), dtype=torch.float32)
mat2 = torch.tensor(np.ones(shape=[2, 4, 2]), dtype=torch.float32)
output = torch.bmm(input, mat2).numpy()
print(output)
# [[[4. 4.]]
#  [[4. 4.]]]

# MindSpore
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.ones(shape=[2, 1, 4]), mindspore.float32)
y = Tensor(np.ones(shape=[2, 4, 2]), mindspore.float32)

batmatmul = ops.BatchMatMul()
output = batmatmul(x, y)
print(output)
# [[[4. 4.]]
#  [[4. 4.]]]
```

### 代码示例2

pytorch只支持3D的tensor，MindSpore支持3D以及更高维度的矩阵乘法计算。

```python
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.ones(shape=[3, 5, 1, 3]), mindspore.float32)
y = Tensor(np.ones(shape=[3, 5, 3, 4]), mindspore.float32)

batmatmul = ops.BatchMatMul()
output = batmatmul(x, y)
print(output.shape)
# (3, 5, 1, 4)
```

### 代码示例3

MindSpore的transpose_a若为True，会把输入相乘的第一个tensor的最后两维进行交换，transpose_b若为True，会把输入相乘的第二个tensor的最后两维进行交换。

```python
import numpy as np
import mindspore
import mindspore.ops as ops
from mindspore import Tensor

x = Tensor(np.ones(shape=[3, 5, 3, 1]), mindspore.float32)
y = Tensor(np.ones(shape=[3, 5, 3, 4]), mindspore.float32)

batmatmul = ops.BatchMatMul(transpose_a=True)
output = batmatmul(x, y)
print(output.shape)
# (3, 5, 1, 4)
```

