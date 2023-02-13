# 比较与torch.Bilinear的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/BiDense.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Bilinear

```text
torch.nn.Bilinear(
    in1_features,
    in2_features,
    out_features,
    bias=True,
    device=None,
    dtype=None)(input1, input2) -> Tensor
```

更多内容详见[torch.nn.Bilinear](https://pytorch.org/docs/1.8.1/generated/torch.nn.Bilinear.html#torch.nn.Bilinear)。

## mindspore.nn.BiDense

```text
mindspore.nn.BiDense(
    in1_channels,
    in2_channels,
    out_channels,
    weight_init=None,
    bias_init=None,
    has_bias=True)(input1, intput2) -> Tensor
```

更多内容详见[mindspore.nn.BiDense](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.BiDense.html#mindspore.nn.BiDense)。

## 差异对比

PyTorch：双线性全连接层。

MindSpore：MindSpore此API实现功能与PyTorch基本一致。可以通过 `weight_init` 和 `bias_init` 分别设置权重和偏置的初始化方法，PyTorch无此功能。

| 分类 | 子类  | PyTorch      | MindSpore    | 差异                          |
| ---- | ----- | ------------ | ------------ | ---------------------------- |
| 参数 | 参数1 | in1_features | in1_channels  | 功能一致，参数名不同          |
|      | 参数2 | in2_features | in2_channels | 功能一致，参数名不同           |
|      | 参数3 | out_features | out_channels     | 功能一致，参数名不同       |
|      | 参数4 | - | weight_init  | 权重参数的初始化方法，PyTorch无此参数      |
|      | 参数5 | - | bias_init    | 偏置参数的初始化方法，PyTorch无此参数      |
|      | 参数6 | bias | has_bias   |   功能一致，参数名不同                   |
|  输入 | 输入1 | input1 | input1 | 功能一致  |
|  输入 | 输入2 | input2 | input2 | 功能一致  |

### 代码示例

```python
# PyTorch
import torch

m = torch.nn.Bilinear(20, 30, 40)
input1 = torch.randn(128, 20)
input2 = torch.randn(128, 30)
output = m(input1, input2)
print(output.shape)
# torch.Size([128, 40])

# MindSpore
import mindspore
m = mindspore.nn.BiDense(20, 30, 40)
input1 = mindspore.ops.randn(128, 20)
input2 = mindspore.ops.randn(128, 30)
output = m(input1, input2)
print(output.shape)
# (128, 40)
```
