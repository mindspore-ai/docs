# 比较与torch.nn.CosineEmbeddingLoss的功能差异

## torch.nn.CosineEmbeddingLoss

```text
class torch.nn.CosineEmbeddingLoss(
    margin=0.0,
    reduction='mean',
    size_average=None,
    reduce=None
)(logits_x1, logits_x2, labels) -> Number
```

更多内容详见 [torch.nn.CosineEmbeddingLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CosineEmbeddingLoss.html)。

## mindspore.nn.CosineEmbeddingLoss

```text
class mindspore.nn.CosineEmbeddingLoss(
    margin=0.0,
    reduction='mean'
)(logits_x1, logits_x2, labels) -> Number
```

更多内容详见 [mindspore.nn.CosineEmbeddingLoss](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.CosineEmbeddingLoss.html)。

## 差异对比

PyTorch：余弦相似度损失函数，用于测量两个Tensor之间的相似性

MindSpore：与PyTorch实现同样的功能

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
|参数 | 参数1 | margin    | margin | - |
| | 参数2 | reduction | reduction | - |
| | 参数3 | logits_x1 | logits_x1 | - |
| | 参数4 | logits_x2 | logits_x2 | - |
| | 参数5 | labels | labels | - |
| | 参数5 | size_average | - | 已弃用 |
| | 参数5 | reduce | - | 已弃用 |

### 代码示例1

> 两API实现功能相同，给定两个Tensor，x1和x2，以及一个Tensor标签y，值为1或-1，使用方法相同

公式如下：

```text
loss(x1,x2,y)={  1−cos(x1,x2),              if y=1

                 max(0,cos(x1,x2)−margin),  if y=−1 }
```

```python
# PyTorch
import torch
from torch import tensor,nn
import numpy as np

logits_x1 = tensor(np.array([[0.3, 0.8], [0.4, 0.3]]))
logits_x2 = tensor(np.array([[0.4, 1.2], [-0.4, -0.9]]))
labels = tensor(np.array([1, -1]))
cosine_embedding_loss = nn.CosineEmbeddingLoss()
output = cosine_embedding_loss(logits_x1, logits_x2, labels)
print(output.detach().numpy())
#0.00034258311711488076

# MindSpore
import mindspore
from mindspore import Tensor,nn
import numpy as np

logits_x1 = Tensor(np.array([[0.3, 0.8], [0.4, 0.3]]), mindspore.float32)
logits_x2 = Tensor(np.array([[0.4, 1.2], [-0.4, -0.9]]), mindspore.float32)
labels = Tensor(np.array([1, -1]), mindspore.int32)
cosine_embedding_loss = nn.CosineEmbeddingLoss()
output = cosine_embedding_loss(logits_x1, logits_x2, labels)
print(output)
#0.0003425479
```
