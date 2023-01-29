# 比较与torch.nn.GRU的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/GRU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.nn.GRU

```text
torch.nn.GRU(*args, **kwargs)(input, h_0) -> Tensor
```

更多内容详见[torch.nn.GRU](https://pytorch.org/docs/1.8.1/generated/torch.nn.GRU.html#torch.nn.GRU)。

## mindspore.nn.GRU

```text
mindspore.nn.GRU(*args, **kwargs)(x, hx, seq_length)  -> Tensor
```

更多内容详见[mindspore.nn.GRU](https://mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.GRU.html)。

## 差异对比

PyTorch：根据输出序列和给定的初始状态计算输出序列和最终状态。

MindSpore：功能一致，仅参数名不同。

| 分类 | 子类   | PyTorch       | MindSpore     | 差异                      |
| ---- | ------ | :------------ | ------------- | ------------------------- |
| 参数 | 参数1  | input_size    | input_size    | -                         |
|      | 参数2  | hidden_size   | hidden_size   | -                         |
|      | 参数3  | num_layers    | num_layers     | -                         |
|      | 参数4  | bias          | has_bias      | 功能一致，参数名不同      |
|      | 参数5  | batch_first   | batch_first   | -                         |
|      | 参数6  | dropout       | dropout       | -                         |
|      | 参数7  | bidirectional | bidirectional | -                         |
|      | 参数8  | input         | x             | 功能一致，参数名不同      |
|      | 参数9  | h_0           | hx            | 功能一致，参数名不同      |
|      | 参数10 | -             | seq_length    | 输入batch中每个序列的长度 |

### 代码示例1

> 两API实现功能一致，用法相同。

```python
# PyTorch
import torch
import torch.nn as nn
import numpy as np

rnn = nn.GRU(10, 16, 2, batch_first=True)
input = torch.ones([3,5,10], dtype=torch.float32)
h0 = torch.ones([1 * 2,3,16], dtype=torch.float32)
output, hn = rnn(input, h0)
output = output.detach().numpy()
print(output.shape)
# (3, 5, 16)

# MindSpore
from mindspore import Tensor, nn

net = nn.GRU(10, 16, 2, batch_first=True)
x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
output, hn = net(x, h0)
print(output.shape)
# (3, 5, 16)
```
