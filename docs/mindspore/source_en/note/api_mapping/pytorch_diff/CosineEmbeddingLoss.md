# Function Differences with torch.nn.CosineEmbeddingLoss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/CosineEmbeddingLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.CosineEmbeddingLoss

```text
class torch.nn.CosineEmbeddingLoss(
    margin=0.0,
    size_average=None,
    reduce=None,
    reduction='mean'
)(x1, x2, target) -> Tensor/Scalar
```

For more information, see [torch.nn.CosineEmbeddingLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CosineEmbeddingLoss.html).

## mindspore.nn.CosineEmbeddingLoss

```text
class mindspore.nn.CosineEmbeddingLoss(
    margin=0.0,
    reduction='mean'
)(logits_x1, logits_x2, labels) -> Tensor/Scalar
```

For more information, see [mindspore.nn.CosineEmbeddingLoss](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.CosineEmbeddingLoss.html).

## Differences

PyTorch: Cosine similarity loss function for measuring the similarity between two Tensors.

MindSpore: Implement the same function as PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | --- | --- | --- |---|
|Parameters | Parameter 1 | margin    | margin | - |
| | Parameter 2 | size_average | - | Deprecated, function taken over by reduction |
| | Parameter 3 | reduce | - | Deprecated, function taken over by reduction |
| | Parameter 4 | reduction | reduction | - |
| | Parameter 5 | x1 | logits_x1 |  Same function, different parameter names  |
| | Parameter 6 | x2 | logits_x2 |  Same function, different parameter names  |
| | Parameter 7 | target | labels |  Same function, different parameter names  |

### Code Example 1

> The two APIs achieve the same function and have the same usage.

```python
# PyTorch
import torch
from torch import tensor, nn
import numpy as np

input1 = tensor(np.array([[0.3, 0.8], [0.4, 0.3]]))
input2 = tensor(np.array([[0.4, 1.2], [-0.4, -0.9]]))
target = tensor(np.array([1, -1]))
cosine_embedding_loss = nn.CosineEmbeddingLoss()
output = cosine_embedding_loss(input1, input2, target)
print(output.detach().numpy())
# 0.00034258311711488076

# MindSpore
import mindspore
from mindspore import Tensor, nn
import numpy as np

logits_x1 = Tensor(np.array([[0.3, 0.8], [0.4, 0.3]]), mindspore.float32)
logits_x2 = Tensor(np.array([[0.4, 1.2], [-0.4, -0.9]]), mindspore.float32)
labels = Tensor(np.array([1, -1]), mindspore.int32)
cosine_embedding_loss = nn.CosineEmbeddingLoss()
output = cosine_embedding_loss(logits_x1, logits_x2, labels)
print(output)
# 0.0003425479
```
