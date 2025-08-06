# 比较与torchvision.transforms.RandomResizedCrop的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomResizedCrop.md)

## torchvision.transforms.RandomResizedCrop

```python
class torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=InterpolationMode.BILINEAR)
```

更多内容详见[torchvision.transforms.RandomResizedCrop](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomResizedCrop)。

## mindspore.dataset.vision.RandomResizedCrop

```python
class mindspore.dataset.vision.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Inter.BILINEAR, max_attempts=10)
```

更多内容详见[mindspore.dataset.vision.RandomResizedCrop](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset_vision/mindspore.dataset.vision.RandomResizedCrop.html)。

## 差异对比

PyTorch：对输入图像进行随机裁剪，并使用指定的插值方式将图像调整为指定的尺寸大小。

MindSpore：对输入图像进行随机裁剪，并使用指定的插值方式将图像调整为指定的尺寸大小。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | size    | size    | - |
|     | 参数2 | scale      | scale   |- |
|     | 参数3 | ratio     | ratio    | - |
|     | 参数4 | interpolation     | interpolation   | - |
|     | 参数5 | -     | max_attempts   | 生成随机裁剪位置的最大尝试次数，超过该次数时将使用中心裁剪 |

## 代码示例

```python
from download import download
from PIL import Image

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

op = T.RandomResizedCrop((32, 32), scale=(0.08, 1.0), ratio=(0.75, 0.8))
img_torch =op(orig_img)
print(img_torch.size)
# Out: (32, 32)

# MindSpore
import mindspore.dataset.vision as vision

op = vision.RandomResizedCrop((32, 32), scale=(0.08, 1.0), ratio=(0.75, 0.8))
img_ms = op(orig_img)
print(img_ms.size)
# Out: (32, 32)
```