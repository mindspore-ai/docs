# 比较与torch.nn.Dropout的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.11/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Dropout.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png"></a>

## torch.nn.Dropout

```python
torch.nn.Dropout(p=0.5, inplace=False)
```

更多内容详见[torch.nn.Dropout](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout.html)。

## mindspore.nn.Dropout

```python
mindspore.nn.Dropout(keep_prob=0.5, p=None)
```

更多内容详见[mindspore.nn.Dropout](https://mindspore.cn/docs/zh-CN/r1.11/api_python/nn/mindspore.nn.Dropout.html)。

## 差异对比

PyTorch：Dropout是一种正则化手段，该算子根据丢弃概率 `p` ，在训练过程中随机将一些神经元输出设置为0，通过阻止神经元节点间的相关性来减少过拟合。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。`keep_prob` 是输入神经元保留率，现已废弃。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                                         |
| ---- | ----- | ------- | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 |        | keep_prob         | MindSpore废弃参数  |
|      | 参数2 | p |  p  |  参数名一致，功能一致   |
|      | 参数3 | inplace |  -  | MindSpore无此参数 |

### 代码示例

> 当inplace输入为False时，两API实现相同的功能。

```python
# PyTorch
import torch
from torch import tensor
input = tensor([[1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
                [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
                [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
                [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
                [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]])
output = torch.nn.Dropout(p=0.2, inplace=False)(input)
print(output.shape)
# torch.Size([5, 10])

# MindSpore
import mindspore
from mindspore import Tensor
x = Tensor([[1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]], mindspore.float32)
output = mindspore.nn.Dropout(p=0.2)(x)
print(output.shape)
# (5, 10)
```