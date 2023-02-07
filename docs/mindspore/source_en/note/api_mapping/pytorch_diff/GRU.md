# Function Differences with torch.nn.GRU

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/GRU.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.GRU

```text
class torch.nn.GRU(
    input_size,
    hidden_size,
    num_layers=1,
    bias=True,
    batch_first=False,
    dropout=0,
    bidirectional=False)(input, h_0) -> Tensor
```

For more information, see [torch.nn.GRU](https://pytorch.org/docs/1.8.1/generated/torch.nn.GRU.html).

## mindspore.nn.GRU

```text
class mindspore.nn.GRU(
    input_size,
    hidden_size,
    num_layers=1,
    has_bias=True,
    batch_first=False,
    dropout=0.0,
    bidirectional=False)(x, hx, seq_length) -> Tensor
```

For more information, see [mindspore.nn.GRU](https://mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.GRU.html).

## Differences

PyTorch: Calculate the output sequence and the final state based on the output sequence and the given initial state.

MindSpore: Consistent function. One more interface input seq_length, indicating the length of each sequence in the input batch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ------ | :------------ | ------------- | ------------------------- |
| Parameters | Parameter 1  | input_size| input_size  | - |
|      | Parameter 2  | hidden_size   | hidden_size | -  |
|      | Parameter 3  | num_layers    | num_layers   | -   |
|      | Parameter 4  | bias          | has_bias      | Same function, different parameter names      |
|      | Parameter 5  | batch_first   | batch_first   | -                         |
|      | Parameter 6  | dropout       | dropout       | -                         |
|      | Parameter 7  | bidirectional | bidirectional | -                         |
|   Input   | Input 1  | input         | x             | Same function, different parameter names      |
|      | Input 2 | h_0           | hx            | Same function, different parameter names      |
|      | Input 3 | -             | seq_length    |  The length of each sequence in input batch |

### Code Example

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
import torch.nn as nn
import numpy as np

rnn = nn.GRU(10, 16, 2, batch_first=True)
input = torch.ones([3, 5, 10], dtype=torch.float32)
h0 = torch.ones([1 * 2, 3, 16], dtype=torch.float32)
output, hn = rnn(input, h0)
output = output.detach().numpy()
print(output.shape)
# (3, 5, 16)

# MindSpore
import mindspore
from mindspore import Tensor, nn

net = nn.GRU(10, 16, 2, batch_first=True)
x = Tensor(np.ones([3, 5, 10]), mindspore.float32)
h0 = Tensor(np.ones([1 * 2, 3, 16]), mindspore.float32)
output, hn = net(x, h0)
print(output.shape)
# (3, 5, 16)
```
