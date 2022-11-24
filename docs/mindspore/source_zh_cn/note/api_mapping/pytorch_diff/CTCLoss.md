# 比较与torch.nn.CTCLoss的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/CTCLoss.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.CTCLoss

```text
class torch.nn.CTCLoss(
    blank=0,
    reduction='mean',
    zero_infinity=False
)(inputs, targets, input_lengths, target_lengths) -> Tensor
```

更多内容详见 [torch.nn.CTCLoss](https://pytorch.org/docs/1.8.1/generated/torch.nn.CTCLoss.html)。

## mindspore.ops.CTCLoss

```text
class mindspore.ops.CTCLoss(
    preprocess_collapse_repeated=False,
    ctc_merge_repeated=True,
    ignore_longer_outputs_than_inputs=False
)(x, labels_indices, labels_values, sequence_length) -> (Tensor, Tensor)
```

更多内容详见 [mindspore.ops.CTCLoss](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.CTCLoss.html)。

## 差异对比

PyTorch: 计算连续时间序列和目标序列之间的损失。`loss`的形状和`reduction`的设置有关，若指定为`mean`（默认）或者`sum`，则返回一个标量，若指定为'none'，则返回`batchsize`个标量。

MindSpore: MindSpore此API实现功能与PyTorch基本一致，返回`loss`及其`梯度`。

| 分类 | 子类   | PyTorch             | MindSpore                         | 差异                                                         |
| ---- | ------ | ------------------- | --------------------------------- | ------------------------------------------------------------ |
| 参数 | 参数1  | blank=0             |          -                         | MindSpore无此参数。空白标签为 num_classes - 1     |
|      | 参数2  | reduction='mean'    |           -                        | MindsSpore无此参数。MindSpore默认不对损失结果进行处理 |
|      | 参数3  | zero_infinity=False | ignore_longer_outputs_than_inputs | 功能一致，参数名称不同。用于解决输入序列长度小于输出序列长度的问题 |
|      | 参数4  | inputs               | x                                 | 功能一致，参数名称不同                                       |
|      | 参数5  | targets              | labels_values                     | 功能一致，参数名称不同                                       |
|      | 参数6  | input_lengths       |        -                           | MindSpore无此参数，PyTorch用此参数控制输入各个批次的长度                                |
|      | 参数7  | target_lengths      | sequence_length                   | 功能一致，参数名称不同                                       |
|      | 参数8  |    -                 | preprocess_collapse_repeated      | 如果为True，在CTC计算之前将折叠重复标签。默认值：False       |
|      | 参数9  |    -                 | ctc_merge_repeated                | 如果为False，在CTC计算过程中，重复的非空白标签不会被合并，这些标签将被解释为单独的标签。这是CTC的简化版本。默认值：True |
|      | 参数10 |      -               | labels_indices                    | labels_indices[i, :] = [b, t] 表示 labels_values[i] 存储 (batch b, time t) 的ID，保证了labels_values的秩为1 |

### 代码示例

> 功能一致

```python
# PyTorch
import torch
# Target are to be padded
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 2      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes
# Initialize random batch of input vectors, for *size = (T,N,C)
input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
ctc_loss = torch.nn.CTCLoss(reduction='none')
loss = ctc_loss(input, target, input_lengths, target_lengths)
print(loss.detach().numpy().shape)
# Out:
# (2,)


# MindSpore
import mindspore
from mindspore import Tensor, ops
import numpy as np

x = Tensor(np.array([[[0.56352055, -0.24474338, -0.29601783], [0.8030011, -1.2187808, -0.6991761]], [[-0.81990826, -0.3598757, 0.50144005], [-1.0980303, 0.60394925, 0.3771529]]]).astype(np.float32))

labels_indices = Tensor(np.array([[0, 1], [1, 1]]), mindspore.int64)
labels_values = Tensor(np.array([0, 1]), mindspore.int32)
sequence_length = Tensor(np.array([2, 2]), mindspore.int32)

ctc_loss = ops.CTCLoss()
loss, gradient = ctc_loss(x, labels_indices, labels_values, sequence_length)
print(loss.shape)
# Out:
# (2,)

```
