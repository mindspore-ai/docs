# 比较与torch.nn.Sequential的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SequentialCell.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torch.nn.Sequential

```python
torch.nn.Sequential(
    *args
)
```

更多内容详见[torch.nn.Sequential](https://pytorch.org/docs/1.8.1/generated/torch.nn.Sequential.html)。

## mindspore.nn.SequentialCell

```python
mindspore.nn.SequentialCell(
    *args
)
```

更多内容详见[mindspore.nn.SequentialCell](https://www.mindspore.cn/docs/zh-CN/master/api_python/nn/mindspore.nn.SequentialCell.html)。

## 差异对比

PyTorch: 构造Cell顺序容器。Sequential按照传入List的顺序依次将Cell添加。此外，也支持OrderedDict作为构造器传入。

MindSpore: 构造Cell顺序容器。入参类型和PyTorch一致。和PyTorch相比，MindSpore支持append()，在容器末尾添加Cell。

| 分类 | 子类  | PyTorch      | MindSpore | 差异                                                         |
| ---- | ----- | ------------ | --------- | ------------------------------------------------------------ |
| 参数| 参数1 | args |  args  | 传入容器的参数，支持List和OrderedDict类型。 |

## 代码示例

```python
import collections

# In MindSpore
import mindspore as ms

model = ms.nn.SequentialCell(
          ms.nn.Conv2d(1,20,5),
          ms.nn.ReLU(),
          ms.nn.Conv2d(20,64,5),
          ms.nn.ReLU()
        )
print(model)
# Out:
# SequentialCell<
#   (0): Conv2d<input_channels=1, output_channels=20, kernel_size=(5, 5), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#   (1): ReLU<>
#   (2): Conv2d<input_channels=20, output_channels=64, kernel_size=(5, 5), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#   (3): ReLU<>
#   >

# Example of using Sequential with OrderedDict
model = ms.nn.SequentialCell(collections.OrderedDict([
          ('conv1', ms.nn.Conv2d(1,20,5)),
          ('relu1', ms.nn.ReLU()),
          ('conv2', ms.nn.Conv2d(20,64,5)),
          ('relu2', ms.nn.ReLU())
        ]))
print(model)
# Out:
# SequentialCell<
#   (conv1): Conv2d<input_channels=1, output_channels=20, kernel_size=(5, 5), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#   (relu1): ReLU<>
#   (conv2): Conv2d<input_channels=20, output_channels=64, kernel_size=(5, 5), stride=(1, 1), pad_mode=same, padding=0, dilation=(1, 1), group=1, has_bias=False, weight_init=normal, bias_init=zeros, format=NCHW>
#   (relu2): ReLU<>
#   >


# In PyTorch
import torch

model = torch.nn.Sequential(
          torch.nn.Conv2d(1,20,5),
          torch.nn.ReLU(),
          torch.nn.Conv2d(20,64,5),
          torch.nn.ReLU()
        )
print(model)
# Out
# Sequential(
#   (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (1): ReLU()
#   (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#   (3): ReLU()
# )

# Example of using Sequential with OrderedDict
model = torch.nn.Sequential(collections.OrderedDict([
          ('conv1', torch.nn.Conv2d(1,20,5)),
          ('relu1', torch.nn.ReLU()),
          ('conv2', torch.nn.Conv2d(20,64,5)),
          ('relu2', torch.nn.ReLU())
        ]))
print(model)
# Out：
# Sequential(
#   (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (relu1): ReLU()
#   (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#   (relu2): ReLU()
# )
```
