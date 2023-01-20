# Function Differences with torch.flatten

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/flatten.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.flatten

```python
torch.flatten(
    input,
    start_dim=0,
    end_dim=-1
)
```

For more information, see [torch.flatten](https://pytorch.org/docs/1.8.1/generated/torch.flatten.html).

## mindspore.ops.flatten

```python
mindspore.ops.flatten(input_x)
```

For more information,
see [mindspore.ops.flatten](https://mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.flatten.html).

## Differences

PyTorch: Supports the flatten operation of elements by specified dimensions.

MindSpore：Only the 0th dimension element is reserved and the elements of the remaining dimensions are flattened.

| Categories | Subcategories | PyTorch   | MindSpore | Differences                                                   |
|------------|---------------|-----------|-----------|---------------------------------------------------------------|
| Parameter  | Parameter 1   | input     | input_x   | The function is the same, and the parameter name is different |
|            | Parameter 2   | start_dim | -         | MindSpore does not have this Parameter                        |
|            | Parameter 3   | end_dim   | -         | MindSpore does not have this Parameter                        |

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, only the 0th dimension will be reserved and the rest will be flattened.
input_tensor = ms.Tensor(np.ones(shape=[1, 2, 3, 4]), ms.float32)
output = ops.flatten(input_tensor)
print(output.shape)
# Out：
# (1, 24)

# In torch, the dimension to reserve will be specified and the rest will be flattened.
input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
output1 = torch.flatten(input=input_tensor, start_dim=1)
print(output1.shape)
# Out：
# torch.Size([1, 24])

input_tensor = torch.Tensor(np.ones(shape=[1, 2, 3, 4]))
output2 = torch.flatten(input=input_tensor, start_dim=2)
print(output2.shape)
# Out：
# torch.Size([1, 2, 12])
```