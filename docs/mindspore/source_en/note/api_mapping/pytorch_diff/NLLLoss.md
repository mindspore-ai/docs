# Function Differences with torch.nn.NLLLoss

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/NLLLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.NLLLoss

```python
torch.nn.NLLLoss(
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduce=None,
    reduction='mean'
)
```

For more information, see [torch.nn.NLLLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.NLLLoss.html).

## mindspore.nn.NLLLoss

```python
class mindspore.nn.NLLLoss(
    weight=None,
    ignore_index=-100,
    reduction='mean'
)(logits, labels)
```

For more information, see [mindspore.nn.NLLLoss](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.NLLLoss.html).

## Differences

PyTorch: Calculate the negative log-likelihood loss between the predicted and target values.

MindSpore: There are no functional differences except for two parameters that have been deprecated in PyTorch.

| Categories | Subcategories |PyTorch | MindSpore | Differences |
| --- | --- | --- | --- |---|
| Parameters | Parameter 1 | weight | weight    | Specify the weight of each category |
| | Parameter 2 | size_average | -         | Deprecated, replaced by reduction. MindSpore does not have this parameter |
| | Parameter 3 | ignore_index | ignore_index | Specify the values to be ignored in the labels (generally padding values) so that they do not have an effect on the gradient |
| | Parameter 4 | reduce | - | Deprecated, replaced by reduction. MindSpore does not have this parameter |
| | Parameter 5 | reduction         | reduction      | Specify the calculation method to be applied to the output results |

## Code Example

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
