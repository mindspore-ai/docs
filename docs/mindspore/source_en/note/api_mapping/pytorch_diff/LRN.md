# # Differences with torch.nn.LocalResponseNorm

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/LRN.md)

## torch.nn.LocalResponseNorm

```text
class torch.nn.LocalResponseNorm(
    size,
    alpha=0.0001,
    beta=0.75,
    k=1.0
)(input) -> Tensor
```

For more information, see [torch.nn.LocalResponseNorm](https://pytorch.org/docs/1.8.1/generated/torch.nn.LocalResponseNorm.html).

## mindspore.nn.LRN

```text
class mindspore.nn.LRN(
    depth_radius=5,
    bias=1.0,
    alpha=1.0,
    beta=0.5,
    norm_region="ACROSS_CHANNELS"
)(x) -> Tensor
```

For more information, see [mindspore.nn.LRN](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.LRN.html).

## Differences

PyTorch: This API performs Local Response Normalization (LRN) operation that normalizes the input for each neuron in a specific way to improve the generalization ability of deep neural networks. It returns a tensor with the same type as the input.

MindSpore: It implements the same functionality as PyTorch, but with different parameter names. The `depth_radius` parameter in MindSpore performs the same function as the `size` parameter in PyTorch, and there is a mapping relationship of twice the value: size=2*depth_radius. Currently, mindspore.nn.LRN and tf.raw_ops.LRN can be completely aligned, and both can achieve the same accuracy. However, if compared with torch.nn.LocalResponseNorm, there may be a precision difference of 1e-3.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameters 1 | size       | depth_radius         | The number of adjacent neurons to consider for normalization,mapping relationship: size=2*depth_radius|
| | Parameter 2 | k       | bias         | Same function, different parameter names      |
| | Parameter 3 | alpha       | alpha         | - |
| | Parameter 4 | beta       | beta         | - |
| | Parameter 5 | -       | norm_region         | Specify the norm region, PyTorch doesn't have this parameter |
| Input | Single input | input  | x | Same function, different parameter names |

### Code Example 1

> The`depth_radius` in MindSpore corresponds to `size` in PyTorch with its value halved, it is necessary to set `depth_radius` as half of `size` to achieve the same function.

```python
# PyTorch
import torch
import numpy as np

input_x = torch.from_numpy(np.array([[[[2.4], [3.51]],[[1.3], [-4.4]]]], dtype=np.float32))
output = torch.nn.LocalResponseNorm(size=2, alpha=0.0001, beta=0.75, k=1.0)(input_x)
print(output.numpy())
#[[[[ 2.3994818]
#   [ 3.5083795]]
#  [[ 1.2996368]
#   [-4.39478  ]]]]

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

input_x = Tensor(np.array([[[[2.4], [3.51]],[[1.3], [-4.4]]]]), mindspore.float32)
lrn = mindspore.nn.LRN(depth_radius=1, bias=1.0, alpha=0.0001, beta=0.75)
output = lrn(input_x)
print(output)
#[[[[ 2.39866  ]
#   [ 3.5016835]]
#  [[ 1.2992741]
#   [-4.3895745]]]]
```
