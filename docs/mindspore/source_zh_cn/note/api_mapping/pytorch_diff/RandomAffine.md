# 比较与torchvision.transforms.RandomAffine的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomAffine.md)

## torchvision.transforms.RandomAffine

```python
class torchvision.transforms.RandomAffine(degrees, translate=None, scale=None, shear=None, interpolation=<InterpolationMode.NEAREST: 'nearest'>, fill=0, fillcolor=None, resample=None)
```

更多内容详见[torchvision.transforms.RandomAffine](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomAffine)。

## mindspore.dataset.vision.RandomAffine

```python
class mindspore.dataset.vision.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=Inter.NEAREST, fill_value=0)
```

更多内容详见[mindspore.dataset.vision.RandomAffine](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset_vision/mindspore.dataset.vision.RandomAffine.html)。

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
|     | 参数7 | fillcolor   | -  | PyTorch已废弃此参数，与fill参数功能相同 |
|     | 参数8 | resample    | -  | PyTorch已废弃此参数，与interpolation参数功能相同 |

## 代码示例

```python
from download import download
from PIL import Image

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

affine_transfomer = T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), fill=0)
img_torch = affine_transfomer(orig_img)

# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

affine_transfomer = vision.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), fill_value=0)
img_ms = affine_transfomer(orig_img)
```
