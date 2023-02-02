# Function Differences with torch.nn.LSTMCell

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/LSTMCell.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.LSTMCell

```text
class torch.nn.LSTMCell(
    input_size,
    hidden_size,
    bias=True)(input, h_0, c_0) -> Tensor
```

For more information, see [torch.nn.LSTMCell](https://pytorch.org/docs/1.8.1/generated/torch.nn.LSTMCell.html?torch.nn.LSTMCell).

## mindspore.nn.LSTMCell

```text
class mindspore.nn.LSTMCell(
    input_size,
    hidden_size,
    has_bias=True)(x, hx) -> Tensor
```

For more information, see [mindspore.nn.LSTMCell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.LSTMCell.html).

## Differences

PyTorch: Compute long-term and short-term memory network units.

MindSpore: MindSpore API basically implements the same function as PyTorch, but the return value differs in form. h_1 and c_1 are returned in PyTorch, while hx' is returned in MindSpore, which is a tuple of two Tensors (h', c ').

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
|Parameters | Parameter 1 | input_size | input_size |- |
| | Parameter 2 | hidden_size | hidden_size | - |
| | Parameter 3 | bias | has_bias | Same function, different parameter names |
|Inputs | Input 1 | input | x | Same function, different parameter names |
| | Input 2 | h_0 | hx | In MindSpore hx represents a tuple of two Tensor(h_0, c_0), corresponding to inputs 2 and 3 in PyTorch, with the same function  |
| | Input 3 | c_0 | hx | In MindSpore hx represents a tuple of two Tensor(h_0, c_0), corresponding to inputs 2 and 3 in PyTorch, with the same function  |

### Code Example 1

> LSTMCell input dimension is 10, hidden state dimension is 16, hidden layer is 3 rows and 16 columns matrix, cell is 3 rows and 20 columns matrix. 5 for loops compute the whole sequence sequentially, and the (hx,cx) of the current Cell is used as the hidden layer input for the next computation. The two APIs achieve the same function.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

rnn = torch.nn.LSTMCell(10, 16)
input = tensor(np.ones([5, 3, 10]).astype(np.float32))
hx = tensor(np.ones([3, 16]).astype(np.float32))
cx = tensor(np.ones([3, 16]).astype(np.float32))
output = []
for i in range(input.size()[0]):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)
print(tuple(output[0].shape))
# (3, 16)

# MindSpore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

net = nn.LSTMCell(10, 16)
x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
h = Tensor(np.ones([3, 16]).astype(np.float32))
c = Tensor(np.ones([3, 16]).astype(np.float32))
output = []
for i in range(5):
    hx = net(x[i], (h, c))
    output.append(hx)
print(output[0][0].shape)
# (3, 16)
```

### Code Example 2

> When bias=False, without bias b_ih and b_hh. The layer does not use offset weights and the two APIs achieve the same function.

```python
# PyTorch
import torch
from torch import tensor
import numpy as np

rnn = torch.nn.LSTMCell(10, 16, bias=False)
input = tensor(np.ones([5, 3, 10]).astype(np.float32))
hx = tensor(np.ones([3, 16]).astype(np.float32))
cx = tensor(np.ones([3, 16]).astype(np.float32))
output = []
for i in range(input.size()[0]):
    hx, cx = rnn(input[i], (hx, cx))
    output.append(hx)
print(tuple(output[0].shape))
# (3, 16)

# MindSpore
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np

net = nn.LSTMCell(10, 16, has_bias=False)
x = Tensor(np.ones([5, 3, 10]).astype(np.float32))
h = Tensor(np.ones([3, 16]).astype(np.float32))
c = Tensor(np.ones([3, 16]).astype(np.float32))
output = []
for i in range(5):
    hx = net(x[i], (h, c))
    output.append(hx)
print(output[0][0].shape)
# (3, 16)
```
