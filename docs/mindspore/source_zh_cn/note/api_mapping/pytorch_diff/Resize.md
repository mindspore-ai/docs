# 比较与torchvision.transforms.Resize的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Resize.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchvision.transforms.Resize

```python
class torchvision.transforms.Resize(size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None)
```

更多内容详见[torchvision.transforms.Resize](https://pytorch.org/vision/0.14/generated/torchvision.transforms.Resize.html)。

## mindspore.dataset.vision.Resize

```python
class mindspore.dataset.vision.Resize(size, interpolation=Inter.LINEAR)
```

更多内容详见[mindspore.dataset.vision.Resize](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.Resize.html)。

## 差异对比

PyTorch：对张量图像的尺寸进行调整，对BILINEAR、BICUBIC插值类型支持抗锯齿。

MindSpore：对张量图像的尺寸进行调整，不支持限制图像长边尺寸，不支持抗锯齿模式。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | size    | size    | - |
|     | 参数2 | interpolation      | interpolation    |- |
|     | 参数3 | max_size      | -    | 当size为单个int值并发生短边缩放时，会根据长边是否大于max_size做一次长边缩放，此时短边可能会小于size |
|     | 参数4 | antialias    | -   | 对BILINEAR、BICUBIC插值类型支持抗锯齿 |

## 代码示例

```python
from download import download
from PIL import Image

url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

transform = T.Resize(size=(200, 200), interpolation=T.InterpolationMode.BILINEAR)
img_torch = T.Compose([transform])(orig_img)
print(img_torch.size)

# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

transform = vision.Resize(size=(200, 200), interpolation=vision.Inter.BILINEAR)
img_ms = transforms.Compose([transform])(orig_img)
print(img_ms[0].size)
```
