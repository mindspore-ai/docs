# Function Differences with torch.nn.SequentialCell

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SequentialCell.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.nn.Sequential

```python
torch.nn.Sequential(
    *args
)
```

For more information, see [torch.nn.Sequential](https://pytorch.org/docs/1.8.1/generated/torch.nn.Sequential.html).

## mindspore.nn.SequentialCell

```python
mindspore.nn.SequentialCell(
    *args
)
```

For more information, see [mindspore.nn.SequentialCell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.SequentialCell.html).

## Differences

PyTorch: Construct the Cell order container. Sequential adds the Cells in the order of the incoming List. In addition, OrderedDict is also supported as a constructor.

MindSpore: Construct the Cell order container. The input types are the same as PyTorch. In contrast to PyTorch, MindSpore supports append(), which adds the Cell at the end of the container.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| ---- | ----- | ------- | --------- | ------------- |
| Parameter | Parameter 1 | args |  args  | Parameters of the incoming container, supporting List and OrderedDict types. |

## Code Example

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
