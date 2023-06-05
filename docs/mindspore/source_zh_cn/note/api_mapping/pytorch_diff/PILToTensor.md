# 比较与torchvision.transforms.PILToTensor的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/PILToTensor.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchvision.transforms.PILToTensor

```python
class torchvision.transforms.PILToTensor
```

更多内容详见[torchvision.transforms.PILToTensor](https://pytorch.org/vision/0.14/generated/torchvision.transforms.PILToTensor)。

## mindspore.dataset.vision.ToNumpy

```python
class mindspore.dataset.vision.ToNumpy
```

更多内容详见[mindspore.dataset.vision.ToNumpy](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.ToNumpy.html#mindspore.dataset.vision.ToNumpy)。

## 使用方式

PyTorch：将PIL类型的Image转换为torch中的Tensor，输入的Image类型通常是<H, W, C>格式，转换得到的Torch Tensor通常是<C, H, W>格式。

MindSpore：将PIL类型的Image转换为Numpy，输入的Image类型通常是<H, W, C>格式，转换得到的Numpy通常是<C, H, W>格式。

## 代码示例

```python
import numpy as np
from PIL import Image
from torchvision import transforms
import mindspore.dataset.vision as vision

from download import download
from PIL import Image

url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# In torch, PILToTensor transforms the input to tensor.
image_transform = transforms.PILToTensor()
img = Image.open('flamingos.jpg')
img_data = image_transform(img)
print(img_data.shape)
# Out:
# torch.Size([3, 292, 471])

# In MindSpore, ToNumpy convert PIL Image into numpy array.
img = Image.open('flamingos.jpg')
to_numpy = vision.ToNumpy()
img_data = to_numpy(img)
print(img_data.shape)
# Out:
# (292, 471, 3)
```
