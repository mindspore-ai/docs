# 比较与torch.nn.CrossEntropyLoss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/CrossEntropyLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.CrossEntropyLoss

```text
class torch.nn.CrossEntropyLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction='mean'
)(input, target) -> Tensor
```

更多内容详见[torch.nn.CrossEntropyLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CrossEntropyLoss.html)。

## mindspore.nn.CrossEntropyLoss

```text
class mindspore.nn.CrossEntropyLoss(
    weight=None,
    ignore_index=-100,
    reduction='mean',
    label_smoothing=0.0
)(logits, labels) -> Tensor
```

更多内容详见[mindspore.nn.CrossEntropyLoss](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.CrossEntropyLoss.html)。

## 差异对比

PyTorch：计算预测值和目标值之间的交叉熵损失。

MindSpore：MindSpore此API实现功能与PyTorch基本一致，而且目标值支持两种不同的数据形式：标量和概率。

| 分类 | 子类  | PyTorch      | MindSpore       | 差异                                                         |
| ---- | ----- | ------------ | --------------- | ------------------------------------------------------------ |
| 参数 | 参数1 | weight       | weight          | -                                                            |
|      | 参数2 | size_average | - | 已弃用，功能由reduction接替 |
|      | 参数3 | ignore_index | ignore_index    | -                                                            |
|      | 参数4 | reduce | - | 已弃用，功能由reduction接替 |
|      | 参数5 | reduction    | reduction       | -     |
|      | 参数6 |    -     | label_smoothing | 标签平滑值，用于计算Loss时防止模型过拟合的正则化手段。取值范围为[0.0, 1.0]。默认值：0.0 |
| 输入 | 输入1 | input    | logits       | 功能一致，参数名不同    |
|      | 输入2| target    | labels       | 功能一致，参数名不同   |

### 代码示例1

> PyTorch和MindSpore都支持目标值为标量的情况。

```python
# PyTorch
import torch
import numpy as np

input_torch = np.array([[1.624, -0.611, -0.528, -1.072, 0.865], [-2.301, 1.744, -0.761, 0.319, -0.249], [1.462, -2.060, -0.322, -0.384, 1.133]])
target_torch = np.array([1, 0, 4])
loss = torch.nn.CrossEntropyLoss()
input_torch = torch.tensor(input_torch, requires_grad=True)
target_torch = torch.tensor(target_torch, dtype=torch.long)
output = loss(input_torch, target_torch)
print(round(float(output.detach().numpy()), 3))
# 2.764

# MindSpore
import mindspore
import numpy as np

input_ms = np.array([[1.624, -0.611, -0.528, -1.072, 0.865], [-2.301, 1.744, -0.761, 0.319, -0.249], [1.462, -2.060, -0.322, -0.384, 1.133]])
target_ms = np.array([1, 0, 4])
input_ms = mindspore.Tensor(input_ms, mindspore.float32)
target_ms = mindspore.Tensor(target_ms, mindspore.int32)
loss = mindspore.nn.CrossEntropyLoss()
output = loss(input_ms, target_ms)
print(round(float(output), 3))
# 2.764
```