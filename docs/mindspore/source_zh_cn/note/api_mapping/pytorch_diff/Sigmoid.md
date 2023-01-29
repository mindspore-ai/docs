# 比较与torch.nn.Sigmoid的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Sigmoid.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source.png"></a>

## torch.nn.Sigmoid

```text
class torch.nn.Sigmoid()(input) -> Tensor
```

更多内容详见[torch.nn.Sigmoid](https://pytorch.org/docs/1.8.1/generated/torch.nn.Sigmoid.html)。

## mindspore.nn.Sigmoid

```text
class mindspore.nn.Sigmoid()(input_x) -> Tensor
```

更多内容详见[mindspore.nn.Sigmoid](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/nn/mindspore.nn.Sigmoid.html)。

## 差异对比

PyTorch：Sigmoid激活函数，将输入映射到0-1之间。

MindSpore：MindSpore此API实现功能与PyTorch一致，仅实例化后的参数名不同。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| :-: | :-: | :-: | :-: |:-:|
|参数 | 参数1 | input | input_x |功能一致，参数名不同 |

### 代码示例

> 实现功能一致，用法相同。

```python
# PyTorch
import torch
from torch import tensor

input_x = tensor([-1, -2, 0, 2, 1], dtype=torch.float32)
sigmoid = torch.nn.Sigmoid()
output = sigmoid(input_x).numpy()
print(output)
# [0.26894143 0.11920292 0.5        0.880797   0.7310586 ]

# MindSpore
import mindspore
from mindspore import Tensor

input_x = Tensor([-1, -2, 0, 2, 1], mindspore.float32)
sigmoid = mindspore.nn.Sigmoid()
output = sigmoid(input_x)
print(output)
# [0.26894143 0.11920292 0.5        0.8807971  0.7310586 ]
```
