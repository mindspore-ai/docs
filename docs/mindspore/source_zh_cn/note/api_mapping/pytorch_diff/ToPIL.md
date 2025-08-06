# 比较与torchvision.transforms.ToPILImage的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ToPIL.md)

## torchvision.transforms.ToPILImage

```python
class torchvision.transforms.ToPILImage(
    mode=None
    )
```

更多内容详见[torchvision.transforms.ToPILImage](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.ToPILImage)。

## mindspore.dataset.vision.ToPIL

```python
class mindspore.dataset.vision.ToPIL
```

更多内容详见[mindspore.dataset.vision.ToPIL](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset_vision/mindspore.dataset.vision.ToPIL.html#mindspore.dataset.vision.ToPIL)。

## 差异对比

PyTorch：将torch中的Tensor或Numpy数组转换为PIL类型的图像。输入可以是<C, H, W> 格式的torch Tensor，或者<H, W, C> 格式的Numpy数组。

MindSpore：将<H, W, C>格式的Numpy数组（如解码后的图像）转换为PIL图像，不支持指定输入图像的颜色空间。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | mode    | -    | 输入原始数据的颜色空间 |

## 代码示例

```python
import numpy as np
import torch as T
from torchvision.transforms import ToPILImage
import mindspore.dataset.vision as vision

# In MindSpore, ToPIL transform the numpy.ndarray to PIL Image.

image = np.random.random((64,64))
img = vision.ToPIL()(image)
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
