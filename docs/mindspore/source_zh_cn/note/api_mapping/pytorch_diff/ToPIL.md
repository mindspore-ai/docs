# 比较与torchvision.transforms.ToPILImage的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ToPIL.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

## torchvision.transforms.ToPILImage

```python
class torchvision.transforms.ToPILImage(
    mode=None
    )
```

更多内容详见[torchvision.transforms.ToPILImage](https://pytorch.org/vision/0.10/transforms.html#torchvision.transforms.ToPILImage)。

## mindspore.dataset.vision.py_transforms.ToPIL

```python
class mindspore.dataset.vision.py_transforms.ToPIL
```

更多内容详见[mindspore.dataset.vision.py_transforms.ToPIL](https://mindspore.cn/docs/zh-CN/r1.7/api_python/dataset_vision/mindspore.dataset.vision.py_transforms.ToPIL.html#mindspore.dataset.vision.py_transforms.ToPIL)。

## 使用方式

PyTorch：将torch中的Tensor或numpy数组转换为PIL类型的图像。输入可以是<C, H, W> 格式的torch Tensor, 或者<H, W, C> 格式的numpy数组。

MindSpore：输入为解码后的numpy数组，将其转换为PIL类型的图像。

## 代码示例

```python
import numpy as np
import torch as T
from torchvision.transforms import ToPILImage
import mindspore.dataset.vision.py_transforms as py_vision

# In MindSpore, ToPIL transform the numpy.ndarray to PIL Image.

image = np.random.random((64,64))
img = py_vision.ToPIL()(image)
img.show()
# Out:
# window of PIL image

# In torch, ToPILImage transforms the input to PIL Image.
image = T.randn((64, 64))
img = ToPILImage()(image)
img.show()
# Out:
# window of PIL image
```
