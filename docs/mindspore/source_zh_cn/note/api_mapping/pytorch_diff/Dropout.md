# 比较与torch.nn.Dropout的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Dropout.md)

## torch.nn.Dropout

```python
torch.nn.Dropout(p=0.5, inplace=False)
```

更多内容详见[torch.nn.Dropout](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout.html)。

## mindspore.nn.Dropout

```python
mindspore.nn.Dropout(keep_prob=0.5, p=None, dtype=mstype.float32)
```

更多内容详见[mindspore.nn.Dropout](https://mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.Dropout.html)。

## 差异对比

PyTorch：Dropout是一种正则化手段，该算子根据丢弃概率 `p` ，在训练过程中随机将一些神经元输出设置为0，通过阻止神经元节点间的相关性来减少过拟合。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。`keep_prob` 是输入神经元保留率，现已废弃。`dtype` 设置输出Tensor的数据类型，现已废弃。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                                         |
| ---- | ----- | ------- | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 |        | keep_prob         | MindSpore废弃参数  |
|      | 参数2 | p |  p  |  参数名一致，功能一致   |
|      | 参数3 | inplace |  -  | MindSpore无此参数 |
|      | 参数4 | - |  dtype  | MindSpore废弃参数 |

Dropout 常用于防止训练过拟合，有一个重要的 **概率值** 参数，该参数在 MindSpore 中的意义与 PyTorch 和 TensorFlow 中的意义完全相反。

在 MindSpore 中，概率值对应 Dropout 算子的属性 `keep_prob`，是指输入被保留的概率，`1-keep_prob`是指输入被置 0 的概率。

在 PyTorch 和 TensorFlow 中，概率值分别对应 Dropout 算子的属性 `p`和 `rate`，是指输入被置 0 的概率，与 MindSpore.nn.Dropout 中的 `keep_prob` 意义相反。

在PyTorch中，网络默认是训练模式，而MindSpore默认是推理模式，因此默认情况下网络调用Dropout不会生效，会直接返回输入，需要通过 `net.set_train()` 方法将网络调整为训练模式后，才能真正执行Dropout。

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
