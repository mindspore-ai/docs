# # Differences with torch.nn.parameter.Parameter

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Parameter.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg"></a>

## torch.nn.parameter.Parameter

```python
torch.nn.parameter.Parameter(data=None, requires_grad=True)
```

For more details, see [torch.nn.parameter.Parameter](https://pytorch.org/docs/1.8.1/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter).

## mindspore.Parameter

```python
mindspore.Parameter(default_input, name=None, requires_grad=True, layerwise_parallel=False, parallel_optimizer=True)
```

For more details, see [mindspore.Parameter](https://mindspore.cn/docs/en/master/api_python/mindspore/mindspore.Parameter.html#mindspore.Parameter).

## Differences

PyTorch：In PyTorch, there is a special type of tensor known as a "Parameter," which is a subclass of the standard tensor. Unlike regular tensors, Parameters in PyTorch are automatically registered as model parameters, making them subject to updates by optimizers.

MindSpore：In MindSpore, a "Parameter" is also a special type of tensor, but unlike PyTorch, both Parameters and regular tensors inherit from the C interface known as "Tensor_".

Furthermore, there are differences between the `requires_grad` parameter in MindSpore and PyTorch. In PyTorch, this parameter is a backend-level attribute. When set to `False`, it indicates that the gradient does not need to be calculated for the tensor, and it won't be included in the computation graph. It also won't record gradient information for each operation, which can improve computational efficiency in scenarios like inference. In MindSpore, this parameter is a frontend-level attribute. When set to `False`, MindSpore's automatic differentiation mechanism will still compute gradients for the tensor in the backend. It will only affect the way the parameter is presented and used in the frontend. For example, MindSpore's `trainable_params` method will hide attributes with `requires_grad` set to `False`.

Additionally, MindSpore's Parameter has an extra `name` parameter compared to PyTorch. This parameter is strongly associated with the Parameter and is used during graph compilation in the backend or during checkpoints saving. You can specify this parameter manually, but if you don't, MindSpore will automatically name the Parameter.

Finally, when directly printing a MindSpore Parameter, you cannot view the actual values contained inside it. You need to use the `Parameter.value()` method to access the actual values.

| Classification | Subclass  | PyTorch | MindSpore | difference |
| ---- | ----- | ------- | --------- | -------------------- |
| parameter | parameter 1 | data | default_input | Consistent |
| | parameter 2 | - | name | Differences as mentioned above |
| | parameter 3 | requires_grad | requires_grad | Differences as mentioned above |
| | parameter 4 | - | layerwise_parallel | MindSpore-specific parameter related to parallelism, not present in torch |
| | parameter 5 | - | parallel_optimizer | MindSpore-specific parameter related to parallelism, not present in torch |

### Code Example

```python
import numpy as np
from mindspore import Parameter, Tensor

a = Parameter(Tensor(np.ones((1, 2), dtype=np.float32)))
print(a)
# Parameter (name=Parameter, shape=(1, 2), dtype=Float32, requires_grad=True)
print(a.value())
# [[1. 1.]]

import torch

b = torch.nn.parameter.Parameter(torch.tensor(np.ones((1, 2), dtype=np.float32)))
print(b.data)
# tensor([[1., 1.]])
```
