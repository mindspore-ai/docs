# 比较与torchvision.transforms.RandomPerspective的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomPerspective.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchvision.transforms.RandomPosterize

```python
class torchvision.transforms.RandomPosterize(bits, p=0.5)
```

更多内容详见[torchvision.transforms.RandomPosterize](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomPosterize.html)。

## mindspore.dataset.vision.RandomPosterize

```python
class mindspore.dataset.vision.RandomPosterize(bits=(8, 8))
```

更多内容详见[mindspore.dataset.vision.RandomPosterize](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.RandomPosterize.html)。

## 差异对比

PyTorch：随机调整图像的颜色通道的比特位数，可以指定概率值表示是否应用此随机操作。

MindSpore：随机调整图像的颜色通道的比特位数，不支持指定应用变换的概率。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | bits    | bits    | - |
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

transform = T.RandomPosterize(bits=(4, 8))
img_torch = T.Compose([transform])(orig_img)


# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

transform = vision.RandomPosterize(bits=(4, 8))
img_ms = transforms.Compose([transform])(orig_img)
```