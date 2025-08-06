# 比较与torchvision.transforms.RandomRotation的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomRotation.md)

## torchvision.transforms.RandomRotation

```python
class torchvision.transforms.RandomRotation(degrees, interpolation=<InterpolationMode.NEAREST: 'nearest'>, expand=False, center=None, fill=0, resample=None)
```

更多内容详见[torchvision.transforms.RandomRotation](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomRotation)。

## mindspore.dataset.vision.RandomRotation

```python
class mindspore.dataset.vision.RandomRotation(degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0)
```

更多内容详见[mindspore.dataset.vision.RandomRotation](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset_vision/mindspore.dataset.vision.RandomRotation.html)。

## 差异对比

PyTorch：随机旋转输入图像。

MindSpore：随机旋转输入图像。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | degrees  | degrees  | - |
|     | 参数2 | interpolation    | resample  |- |
|     | 参数3 | expand    | expand   |- |
|     | 参数4 | center   | center   | - |
|     | 参数5 | fill   | fill_value  | - |
|     | 参数6 | resample   | - | PyTorch已废弃此参数，与interpolation参数功能相同 |

## 代码示例

```python
from download import download
from PIL import Image

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torch
import torchvision.transforms as T
torch.manual_seed(1)

affine_transfomer = T.RandomRotation(degrees=(30, 70), center=(0, 0))
img_torch = affine_transfomer(orig_img)
print(img_torch.size)
# Out: (471, 292)

# MindSpore
import mindspore as ms
import mindspore.dataset.vision as vision
ms.dataset.config.set_seed(2)

affine_transfomer = vision.RandomRotation(degrees=(30, 70), center=(0, 0))
img_ms = affine_transfomer(orig_img)
print(img_ms.size)
# Out: (471, 292)
```
