# 比较与torchvision.transforms.RandomAffine的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomAffine.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchvision.transforms.RandomAffine

```python
class torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, interpolation=InterpolationMode.NEAREST, fill=0, center=None)
```

更多内容详见[torchvision.transforms.RandomAffine](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomAffine.html)。

## mindspore.dataset.vision.RandomAffine

```python
class mindspore.dataset.vision.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=Inter.NEAREST, fill_value=0)
```

更多内容详见[mindspore.dataset.vision.RandomAffine](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.RandomAffine.html)。

## 差异对比

PyTorch：将张量图像应用随机仿射变换，支持指定旋转中心位置。

MindSpore：对输入图像应用随机仿射变换，旋转中心在图像中心位置，不支持指定旋转中心。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | degrees  | degrees  | - |
|     | 参数2 | translate    | translate  |- |
|     | 参数3 | scale    | scale   |- |
|     | 参数4 | shear   | shear   | - |
|     | 参数5 | interpolation   | resample  | - |
|     | 参数6 | fill   | fill_value | - |
|     | 参数7 | center   | -  | 指定图像的旋转中心 |

## 代码示例

```python
from download import download
from PIL import Image

url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), center=(0, 0))
img_torch = affine_transfomer(orig_img)

# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

# If need to specify center of rotation, use RandomRotation + RandomAffine
rotation_transfomer = vision.RandomRotation(degrees=(30, 70), center=(0, 0))
affine_transfomer = vision.RandomAffine(degrees=(0, 0), translate=(0.1, 0.3))
transformer = transforms.Compose([rotation_transfomer, affine_transfomer])
img_ms = transformer(orig_img)
```
