# 比较与torch.nn.NLLLoss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/NLLLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.NLLLoss

```python
torch.nn.NLLLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction='mean'
)(input, target)
```

更多内容详见[torch.nn.NLLLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.NLLLoss.html)。

## mindspore.nn.NLLLoss

```python
class mindspore.nn.NLLLoss(
    weight=None,
    ignore_index=-100,
    reduction='mean'
)(logits, labels)
```

更多内容详见[mindspore.nn.NLLLoss](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.NLLLoss.html)。

## 差异对比

PyTorch：计算预测值和目标值之间的负对数似然损失。

MindSpore：除两个在PyTorch已弃用的参数不同外，功能上无差异。

| 分类 | 子类  | PyTorch      | MindSpore | 差异                                                         |
| ---- | ----- | ------------ | --------- | ------------------------------------------------------------ |
| 参数| 参数1 | weight | weight    | 指定各类别的权重 |
| | 参数2 | size_average | -         | 已弃用，被reduction取代，MindSpore无此参数 |
| | 参数3 | ignore_index | ignore_index | 指定labels中需要忽略的值(一般为填充值)，使其不对梯度产生影响 |
| | 参数4 | reduce | - | 已弃用，被reduction取代，MindSpore无此参数 |
| | 参数5 | reduction         | reduction      | 指定应用于输出结果的计算方式 |
|  输入 | 输入1 | input | logits | 功能一致，参数名不同 |
|   | 输入2 | target | labels | 功能一致，参数名不同 |

## 代码示例

```python
import numpy as np

data = np.random.randn(2, 2, 3, 3)

# In MindSpore
import mindspore as ms

loss = ms.nn.NLLLoss(ignore_index=-110, reduction="none")
input = ms.Tensor(data, dtype=ms.float32)
target = ms.ops.zeros((2, 3, 3), dtype=ms.int32)
output = loss(input, target)
print(output)
# Out:
# [[[ 0.7047795   0.8196785  -0.7913506 ]
#   [ 0.22157642 -0.18818447 -0.65975004]
#   [ 1.7223285  -0.9269855   0.46461168]]

#  [[ 0.21305805 -2.213903    0.36110482]
#   [-0.1900587  -0.56938815  0.12274747]
#   [ 1.149195   -0.8739661  -1.7944012 ]]]


# In PyTorch
import torch

loss = torch.nn.NLLLoss(ignore_index=-110, reduction="none")
input = torch.tensor(data, dtype=torch.float32)
target = torch.zeros((2, 3, 3), dtype=torch.long)
output = loss(input, target)
print(output)
# Out：
# tensor([[[ 0.7048,  0.8197, -0.7914],
#          [ 0.2216, -0.1882, -0.6598],
#          [ 1.7223, -0.9270,  0.4646]],

#         [[ 0.2131, -2.2139,  0.3611],
#          [-0.1901, -0.5694,  0.1227],
#          [ 1.1492, -0.8740, -1.7944]]])
```
