# Function Differences with torch.nn.AdaptiveAvgPool2d

## torch.nn.AdaptiveAvgPool2d

```python
torch.nn.AdaptiveAvgPool2d(output_size)(input)
```

## mindspore.ops.ReduceMean

```python
class mindspore.ops.ReduceMean(keep_dims=False)(
    input_x,
    axis=()
)
```

## Differences

PyTorch: Applies an adaptive average pooling over the inputs, and the corresponding results are calculated based on the specified output size. It is consistent with the `ReduceMean` of MindSpore only if the output is 1*1.

MindSpore：Computes mean of the given axis.

## Code Example

```python
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, mean of given axis will be computed.
input_x = Tensor(np.random.randn(1, 64, 8, 9).astype(np.float32))
op = ops.ReduceMean(keep_dims=True)
output = op(x=input_x, axis=1)
print(output.shape)
# Out：
# (1, 1, 8, 9)

# In torch, the corresponding results will be returned based on the input shape.
input_x = torch.randn(1, 64, 8, 9)
op = torch.nn.AdaptiveAvgPool2d((5, 7))
output = op(input_x)
print(output.shape)
# Out：
# torch.Size([1, 64, 5, 7])
```

