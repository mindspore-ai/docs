# 比较与torch.nn.RNN的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RNN.md)

## torch.nn.RNN

```text
class torch.nn.RNN(*args, **kwargs)(input, h_0)
```

更多内容详见[torch.nn.RNN](https://pytorch.org/docs/1.8.1/generated/torch.nn.RNN.html)。

## mindspore.nn.RNN

```text
class mindspore.nn.RNN(*args, **kwargs)(x, h_x, seq_length)
```

更多内容详见[mindspore.nn.RNN](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.RNN.html)。

## 差异对比

PyTorch：循环神经网络（RNN）层。

MindSpore：实现与PyTorch一致的功能。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | input_size | input_size   | - |
| | 参数2 | hidden_size | hidden_size |  - |
| | 参数3 | num_layers | num_layers | - |
| | 参数4 | nonlinearity |  nonlinearity | - |
| | 参数5 | bias | has_bias | 功能一致，参数名不同 |
| | 参数6 | batch_first | batch_first | - |
| | 参数7 | dropout | dropout | - |
| | 参数8 | bidirectional | bidirectional | - |
|输入 | 输入1 | input        | x       | 功能一致，参数名不同 |
|      | 输入2 | h_0       | hx      | 功能一致，参数名不同 |
|      | 输入3 | -    | seq_length      | 此输入指明真实的序列长度，以避免使用填充后的元素计算隐藏状态，影响最后的输出。当 x 被填充元素时，建议使用此输入。默认值：None。 |

### 代码示例1

```python
# PyTorch
import torch
from torch import tensor
from torch import nn
import numpy as np

rnn = torch.nn.RNN(2, 3, 4, nonlinearity="relu", bias=False)
x = torch.tensor(np.array([[[3.0, 4.0]]]).astype(np.float32))
h_0 = torch.tensor(np.array([[[1.0, 2.0, 3]], [[3.0, 4.0, 5]], [[3.0, 4.0, 5]], [[3.0, 4.0, 5]]]).astype(np.float32))
output, hx_n = rnn(x, h_0)
print(output)
# tensor([[[0.0000, 0.4771, 0.8548]]], grad_fn=<StackBackward0>)
print(hx_n)
# tensor([[[0.0000, 0.5015, 0.0000]],
#
#        [[2.3183, 0.0000, 1.7400]],
#
#        [[2.0082, 0.0000, 1.4658]],
#
#        [[0.0000, 0.4771, 0.8548]]], grad_fn=<StackBackward0>)

# MindSpore
import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np

rnn = nn.RNN(2, 3, 4, nonlinearity="relu", has_bias=False)
x = Tensor(np.array([[[3.0, 4.0]]]).astype(np.float32))
h_0 = Tensor(np.array([[[1.0, 2.0, 3]], [[3.0, 4.0, 5]], [[3.0, 4.0, 5]], [[3.0, 4.0, 5]]]).astype(np.float32))
output, hx_n = rnn(x, h_0)
print(output)
# [[[2.2204838 0.        2.365325 ]]]
print(hx_n)
#[[[1.4659244  0.         1.3142354 ]]
# [[0.         0.16777739 0.        ]]
# [[3.131722   0.         0.        ]]
# [[2.2204838  0.         2.365325  ]]]
```
