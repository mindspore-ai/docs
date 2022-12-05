# 比较与torch.nn.Dropout的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/dropout.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>
## torch.nn.Dropout

```python
torch.nn.Dropout(p=0.5, inplace=False) -> Tensor
```

更多内容详见[torch.nn.Dropout](https://pytorch.org/docs/1.8.1/generated/torch.nn.Dropout.html?highlight=torch%20nn%20dropout#torch.nn.Dropout)

## mindspore.ops.dropout

```python
mindspore.ops.dropout(x, p=0.5, seed0=0, seed1=0) -> Tensor
```

更多内容详见[mindspore.ops.dropout](https://www.mindspore.cn/docs/zh-CN/master/api_python/ops/mindspore.ops.dropout.html)

## 差异对比

PyTorch：dropout是为了防止或减轻过拟合而使用的函数，它会在不同的训练过程中随机丢弃一部分神经元。也就是以一定的概率p随机将神经元输出设置为0，起到减小神经元相关性的作用。其余未被设置为0的参数将会以$\frac{1}{1-p}$进行缩放。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。

| 分类 | 子类  | PyTorch | MindSpore | 差异                                                         |
| ---- | ----- | ------- | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 | p       | p         | -                                                            |
|      | 参数2 | inplace |           | 如果设置为True，将就地执行此操作，默认值为False。就地执行指在输入本身的内存空间进行操作，即对input也进行Dropout操作并保存。MindSpore无此参数 |
|      | 参数3 |         | x         | (Tensor)，dropout的输入，任意维度的Tensor。                  |
|      | 参数4 |         | seed0     | (int)，算子层的随机种子，用于生成随机数。默认值：0           |
|      | 参数5 |         | seed1     | (int)，全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。默认值：0 |

### 代码示例1

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
from mindspore import ops
from mindspore import Tensor
x = Tensor([[1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00],
            [1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00]], mindspore.float32)
output, mask = ops.dropout(x, p=0.2)
print(output.shape)
# (5, 10)
```