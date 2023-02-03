# Function Differences with torch.nn.LSTM

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/LSTM.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.LSTM

```text
class torch.nn.LSTM(
    input_size,
    hidden_size,
    num_layers=1,
    has_bias=True,
    batch_first=False,
    dropout=0,
    bidirectional=False,
    proj_size=0)(input, (h0, c0)) -> Tensor
```

For more information, see [torch.nn.LSTM](https://pytorch.org/docs/1.8.1/generated/torch.nn.LSTM.html).

## mindspore.nn.LSTM

```text
class mindspore.nn.LSTM(
    input_size,
    hidden_size,
    num_layers=1,
    has_bias=True,
    batch_first=False,
    dropout=0,
    bidirectional=False)(x, hx, seq_length) -> Tensor
```

For more information, see [mindspore.nn.LSTM](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.LSTM.html).

## Differences

PyTorch: Compute the output sequence and final state based on the input sequence and the given initial state.

MindSpore: If the proj_size parameter in PyTorch is not specified, the MindSpore API achieves the same functionality as PyTorch, with only some of the parameter names being different.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1  | input_size    | input_size    | -    |
| | Parameter 2  | hidden_size   | hidden_size   | -     |
| | Parameter 3  | num_layers    | num_layers    | -      |
| | Parameter 4  | bias    | has_bias    | Same function, different parameter names  |
| | Parameter 5  | batch_first   | batch_first   | -       |
| | Parameter 6  | dropout       | dropout       | -      |
| | Parameter 7  | bidirectional | bidirectional | -      |
| | Parameter 8  | proj_size     | -             | In PyTorch, if proj_size>0, the hidden_size in the output shape will become proj_size, and the default value is 0. MindSpore does not have this parameter |
| Inputs | Input 1 | input         | x             | Same function, different parameter names   |
| | Input 2 | h_0           | hx            | In MindSpore hx represents a tuple of two Tensor(h_0, c_0), corresponding to inputs 2 and 3 in PyTorch, with the same function          |
| | Input 3 | c_0           | hx             | In MindSpore hx represents a tuple of two Tensor(h_0, c_0), corresponding to inputs 2 and 3 in PyTorch, with the same function     |
| | Input 4 | -             | seq_length    | This parameter in MindSpore specifies the sequence length of the input batch. PyTorch does not have this parameter               |

### Code Example

> When the parameter proj_size in PyTorch takes the default value of 0, the two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

rnn = torch.nn.LSTM(10, 16, 2, bias=True, batch_first=True, bidirectional=False)
input1 = tensor(np.ones([3, 5, 10]), dtype=torch.float32)
h0 = tensor(np.ones([1 * 2, 3, 16]), dtype=torch.float32)
c0 = tensor(np.ones([1 * 2, 3, 16]), dtype=torch.float32)
output, (hn, cn) = rnn(input1, (h0, c0))
print(output.detach().numpy().shape)
# (3, 5, 16)
print(hn.detach().numpy().shape)
# (2, 3, 16)
print(cn.detach().numpy().shape)
# (2, 3, 16)

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

net = mindspore.nn.LSTM(10, 16, 2, has_bias=True, batch_first=True, bidirectional=False)
x = Tensor(np.ones([3, 5, 10]).astype(np.float32))
h0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
c0 = Tensor(np.ones([1 * 2, 3, 16]).astype(np.float32))
output, (hn, cn) = net(x, (h0, c0))
print(output.shape)
# (3, 5, 16)
print(hn.shape)
# (2, 3, 16)
print(cn.shape)
# (2, 3, 16)
```
