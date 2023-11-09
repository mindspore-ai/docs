# 比较与torch.nn.functional.dropout的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.3/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/drop_out.md)

## torch.nn.functional.dropout

```python
torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False)
```

更多内容详见[torch.nn.functional.dropout](https://pytorch.org/docs/1.8.1/nn.functional.html#torch.nn.functional.dropout)。

## mindspore.ops.dropout

```python
mindspore.ops.dropout(input, p=0.5, training=True, seed=None)
```

更多内容详见[mindspore.ops.dropout](https://mindspore.cn/docs/zh-CN/r2.3/api_python/ops/mindspore.ops.dropout.html)。

## 差异对比

MindSpore此API实现功能与PyTorch基本一致，但由于框架机制不同，入参差异如下：

| 分类 | 子类  | PyTorch | MindSpore | 差异                                                         |
| ---- | ----- | ------- | --------- | ------------------------------------------------------------ |
| 参数 | 参数1 |    input  | input  | 一致  |
|      | 参数2 |    p     |  p     | 一致  |
|      | 参数3 | training  |  training | 一致  |
|      | 参数4 | inplace |  -  | MindSpore无此参数 |
|      | 参数5 |    -    |  seed  | 随机数生成器的种子，PyTorch无此参数 |

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
output = torch.nn.functional.dropout(input)
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
output = mindspore.ops.dropout(x)
print(output.shape)
# (5, 10)
```