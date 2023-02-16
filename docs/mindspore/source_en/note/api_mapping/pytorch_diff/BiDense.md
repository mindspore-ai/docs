# Function Differences with torch.Bilinear

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BiDense.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Bilinear

```text
torch.nn.Bilinear(
    in1_features,
    in2_features,
    out_features,
    bias=True)(input1, input2) -> Tensor
```

For more information, see [torch.nn.Bilinear](https://pytorch.org/docs/1.8.1/generated/torch.nn.Bilinear.html#torch.nn.Bilinear).

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

For more information, see [mindspore.nn.BiDense](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.BiDense.html#mindspore.nn.BiDense).

## Differences

PyTorch: Bilinear fully connected layer.

MindSpore: MindSpore API basically implements the same function as PyTorch. The initialization methods for weights and biases can be set via `weight_init` and `bias_init` respectively, which is not available for PyTorch.

| Categories | Subcategories | PyTorch | MindSpore | Differences   |
| ---- | ----- | ------- | --------- | -------------- |
| Parameters | Parameter 1 | in1_features | in1_channels  | Same function, different parameter names          |
|      | Parameter 2 | in2_features | in2_channels | Same function, different parameter names           |
|      | Parameter 3 | out_features | out_channels     | Same function, different parameter names       |
|      | Parameter 4 | - | weight_init  | Initialization method for the weight parameter, which is not available for PyTorch      |
|      | Parameter 5 | - | bias_init    | Initialization method for the bias parameter, which is not available for PyTorch      |
|      | Parameter 6 | bias | has_bias   |   Same function, different parameter names                   |
|  Inputs | Input 1 | input1 | input1 | Same function  |
|   | Input 2 | input2 | input2 | Same function  |

### Code Example

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
