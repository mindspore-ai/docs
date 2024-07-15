# 比较与torchvision.transforms.ConvertImageDtype的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/TypeCast.md)

## torchvision.transforms.ConvertImageDtype

```python
class torchvision.transforms.ConvertImageDtype(
    dtype: torch.dtype
    )
```

更多内容详见[torchvision.transforms.ConvertImageDtype](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.ConvertImageDtype)。

## mindspore.dataset.transforms.TypeCast

```python
class mindspore.dataset.transforms.TypeCast(
    output_type
    )
```

更多内容详见[mindspore.dataset.transforms.TypeCast](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset_transforms/mindspore.dataset.transforms.TypeCast.html#mindspore.dataset.transforms.TypeCast)。

## 差异对比

PyTorch：将张量图像转换为给定的数据类型并相应缩放值，此算子不支持PIL图像。

MindSpore：将输入的numpy.ndarray图像转换为所需的数据类型。

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

to_tensor = T.ToTensor()
convert = T.ConvertImageDtype(torch.float)
img_torch = T.Compose([to_tensor, convert])((orig_img))
print(img_torch.dtype)
# Out: torch.float32

# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

to_tensor = vision.ToTensor()
convert = transforms.TypeCast("float32")
img_ms = transforms.Compose([to_tensor, convert])((orig_img))
print(img_ms[0].dtype)
# Out: float32
```
