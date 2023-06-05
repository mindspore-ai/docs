# 比较与torchvision.transforms.RandomSolarize的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomSolarize.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchvision.transforms.RandomSolarize

```python
class torchvision.transforms.RandomSolarize(
    threshold,
    p=0.5
    )
```

更多内容详见[torchvision.transforms.RandomSolarize](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomSolarize)。

## mindspore.dataset.vision.RandomSolarize

```python
class mindspore.dataset.vision.RandomSolarize(
    threshold=(0, 255)
    )
```

更多内容详见[mindspore.dataset.vision.RandomSolarize](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.RandomSolarize.html#mindspore.dataset.vision.RandomSolarize)。

## 差异对比

PyTorch：通过反转高于阈值的所有像素值，以给定的概率随机对图像进行曝光操作。

MindSpore：从指定阈值范围内随机选择一个子范围，并将图像像素值调整在子范围内。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        | PyTorch为反转高于阈值的所有像素值，MindSpore为指定阈值范围内随机选择一个子范围进行反转  |
|参数 | 参数1 | threshold   | threshold   | - |
|     | 参数2 | p      | -   | 指定对图像应用变换的概率 |

## 代码示例

```python
from download import download
from PIL import Image

url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

transform = T.RandomSolarize(threshold=128, p=1)
img_torch = T.Compose([transform])(orig_img)


# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

transform = vision.RandomSolarize(threshold=(128, 128))
img_ms = transforms.Compose([transform])(orig_img)


```