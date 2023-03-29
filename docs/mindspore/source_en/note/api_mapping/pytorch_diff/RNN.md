# Function Differences with torch.nn.RNN

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RNN.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png"></a>

## torch.nn.RNN

```text
class torch.nn.RNN(*args, **kwargs)(input, h_0)
```

For more information, see [torch.nn.RNN](https://pytorch.org/docs/1.8.1/generated/torch.nn.RNN.html).

## mindspore.nn.RNN

```text
class mindspore.nn.RNN(*args, **kwargs)(x, h_x, seq_length)
```

For more information, see  [mindspore.nn.RNN](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.RNN.html).

## Differences

PyTorch: Recurrent Neural Network (RNN) layer.

MindSpore: Implement the same functions as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameters | Parameter 1 | input_size | input_size   | - |
| | Parameter 2 | hidden_size | hidden_size |  - |
| | Parameter 3 | num_layers | num_layers | - |
| | Parameter 4 | nonlinearity |  nonlinearity | - |
| | Parameter 5 | bias | has_bias | Same function, different parameter names |
| | Parameter 6 | batch_first | batch_first | - |
| | Parameter 7 | dropout | dropout | - |
| | Parameter 8 | bidirectional | bidirectional | - |
|Inputs | Input 1 | input        | x       | Same function, different parameter names |
|      | Input 2 | h_0       | hx      | Same function, different parameter names |
|      | Input 3 | -    | seq_length      | This input specifies the true sequence length to avoid using the filled elements to calculate the hidden state, which affects the final output. It is recommended to use this input when x is populated with elements. Default value: None. |

### Code Example 1

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
#        [[2.3183, 0.0000, 1.7400]],
#        [[2.0082, 0.0000, 1.4658]],
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
