# 比较与torchvision.transforms.RandomPerspective的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomPerspective.md)

## torchvision.transforms.RandomPerspective

```python
class torchvision.transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=InterpolationMode.BILINEAR, fill=0)
```

更多内容详见[torchvision.transforms.RandomPerspective](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomPerspective.html)。

## mindspore.dataset.vision.RandomPerspective

```python
class mindspore.dataset.vision.RandomPerspective(distortion_scale=0.5, prob=0.5, interpolation=Inter.BICUBIC)
```

更多内容详见[mindspore.dataset.vision.RandomPerspective](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset_vision/mindspore.dataset.vision.RandomPerspective.html)。

## 差异对比

PyTorch：对张量图像应用随机透视变换，可以指定变换区域外的填充像素值。

MindSpore：对输入图像应用随机透视变换，变换区域外的填充像素值为黑色，不支持指定填充颜色。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | distortion_scale   | distortion_scale   | - |
|     | 参数2 | p     | prob   |- |
|     | 参数3 | interpolation     | interpolation    | 默认值不同 |
|     | 参数4 | fill    | -   | 指定变换区域外的填充像素值 |

## 代码示例

```python
from download import download
from PIL import Image

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

transform = T.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=T.InterpolationMode.BILINEAR)
img_torch = T.Compose([transform])(orig_img)


# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

transform = vision.RandomPerspective(distortion_scale=0.5, prob=0.5, interpolation=vision.Inter.BILINEAR)
img_ms = transforms.Compose([transform])(orig_img)
```