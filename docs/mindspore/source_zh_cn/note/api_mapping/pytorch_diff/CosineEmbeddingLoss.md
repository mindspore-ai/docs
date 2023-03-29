# 比较与torch.nn.CosineEmbeddingLoss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/CosineEmbeddingLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.nn.CosineEmbeddingLoss

```text
class torch.nn.CosineEmbeddingLoss(
    margin=0.0,
    size_average=None,
    reduce=None,
    reduction='mean'
)(x1, x2, target) -> Tensor
```

更多内容详见[torch.nn.CosineEmbeddingLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CosineEmbeddingLoss.html)。

## mindspore.nn.CosineEmbeddingLoss

```text
class mindspore.nn.CosineEmbeddingLoss(
    margin=0.0,
    reduction='mean'
)(logits_x1, logits_x2, labels) -> Tensor/Scalar
```

更多内容详见[mindspore.nn.CosineEmbeddingLoss](https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.CosineEmbeddingLoss.html)。

## 差异对比

PyTorch：余弦相似度损失函数，用于测量两个Tensor之间的相似性。

MindSpore：与PyTorch实现同样的功能。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 参数 | 参数1 | margin | margin | - |
| | 参数2 | size_average | - | 已弃用，功能由reduction接替 |
| | 参数3 | reduce | - | 已弃用，功能由reduction接替 |
| | 参数4 | reduction | reduction | - |
| 输入 | 输入1 | x1 | logits_x1 |  功能一致，参数名不同  |
| | 输入2 | x2 | logits_x2 |  功能一致，参数名不同  |
| | 输入3 | target | labels |  功能一致，参数名不同  |

### 代码示例

> 两API实现功能相同，使用方法相同。

```python
# PyTorch
import torch
from torch import tensor, nn
import numpy as np

input1 = tensor(np.array([[0.3, 0.8], [0.4, 0.3]]), dtype=torch.float32)
input2 = tensor(np.array([[0.4, 1.2], [-0.4, -0.9]]), dtype=torch.float32)
target = tensor(np.array([1, -1]), dtype=torch.int32)
cosine_embedding_loss = nn.CosineEmbeddingLoss()
output = cosine_embedding_loss(input1, input2, target)
print(output.detach().numpy())
# 0.0003426075

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
