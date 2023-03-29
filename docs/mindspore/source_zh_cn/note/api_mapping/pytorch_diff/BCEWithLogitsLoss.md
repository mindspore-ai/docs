# 比较与torch.nn.BCEWithLogitsLoss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/BCEWithLogitsLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

## torch.nn.BCEWithLogitsLoss

```text
torch.nn.BCEWithLogitsLoss(
    weight=None,
    size_average=None,
    reduce=None,
    reduction='mean',
    pos_weight=None
)(input, target) -> Tensor
```

更多内容详见[torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.BCEWithLogitsLoss.html)。

## mindspore.nn.BCEWithLogitsLoss

```text
class mindspore.nn.BCEWithLogitsLoss(
    reduction='mean',
    weight=None,
    pos_weight=None
)(logits, labels) -> Tensor
```

更多内容详见[mindspore.nn.BCEWithLogitsLoss](https://mindspore.cn/docs/zh-CN/r2.0/api_python/nn/mindspore.nn.BCEWithLogitsLoss.html)。

## 差异对比

PyTorch：将Sigmoid层和BCELoss组合在一个类中计算预测值和目标值之间的二值交叉熵损失，使其比分开使用Sigmoid后跟BCELoss在数值上更加稳定。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅输入参数名不同。

| 分类 | 子类 | PyTorch | MindSpore | 差异 |
| --- | --- | --- | --- |---|
| 输入 | 输入1 | input | logits | 都是输入Tensor |
| | 输入2 | target | labels | 都是输入Tensor |
| 参数 | 参数1 | weight | weight | 功能一致，参数名相同 |
| | 参数2 | size_average | - | 功能一致，PyTorch已弃用该参数，MindSpore无此参数 |
| | 参数3 | reduce | - | 功能一致，PyTorch已弃用该参数，MindSpore无此参数 |
| | 参数4 | reduction | reduction | 功能一致，参数名相同 |
| | 参数5 | pos_weight | pos_weight | 功能一致，参数名相同 |

### 代码示例1

> 两API实现功能一致，用法相同。PyTorch的BCEWithLogitsLoss算子的三个参数weight、reduction和pos_weight与MindSpore的BCEWithLogitsLoss算子相对应的三个参数功能一致，参数名相同，默认值也相同。默认情况下，MindSpore能得到和PyTorch一样的结果。

```python
# PyTorch
import torch
from torch import Tensor
import numpy as np

np.random.seed(1)
input = Tensor(np.random.rand(1, 2, 3).astype(np.float32))
print(input.numpy())
# [[[4.17021990e-01 7.20324516e-01 1.14374816e-04]
#   [3.02332580e-01 1.46755889e-01 9.23385918e-02]]]
target = Tensor(np.random.randint(2,size=(1, 2, 3)).astype(np.float32))
print(target.numpy())
# [[[0. 1. 1.]
#   [0. 0. 1.]]]
torch_BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()
torch_output = torch_BCEWithLogitsLoss(input, target)
torch_output_np = torch_output.numpy()
print(torch_output_np)
# 0.7142954

# MindSpore
import mindspore
from mindspore import Tensor
import numpy as np

np.random.seed(1)
logits = Tensor(np.random.rand(1, 2, 3).astype(np.float32))
print(logits.asnumpy())
# [[[4.17021990e-01 7.20324516e-01 1.14374816e-04]
#   [3.02332580e-01 1.46755889e-01 9.23385918e-02]]]
labels = Tensor(np.random.randint(2,size=(1, 2, 3)).astype(np.float32))
print(labels.asnumpy())
# [[[0. 1. 1.]
#   [0. 0. 1.]]]
ms_BCEWithLogitsLoss = mindspore.nn.BCEWithLogitsLoss()
ms_output = ms_BCEWithLogitsLoss(logits, labels)
ms_output_np = ms_output.asnumpy()
print(ms_output_np)
# 0.71429545
```
