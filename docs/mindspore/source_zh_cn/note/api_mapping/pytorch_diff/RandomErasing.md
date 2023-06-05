# 比较与torchvision.transforms.RandomErasing的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomErasing.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchvision.transforms.RandomErasing

```python
class torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
```

更多内容详见[torchvision.transforms.RandomErasing](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomErasing.html)。

## mindspore.dataset.vision.RandomErasing

```python
class mindspore.dataset.vision.RandomErasing(prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False, max_attempts=10)
```

更多内容详见[mindspore.dataset.vision.RandomErasing](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.RandomErasing.html)。

## 差异对比

PyTorch：按照指定的概率擦除输入numpy.ndarray图像上随机矩形区域内的像素。

MindSpore：按照指定的概率擦除输入numpy.ndarray图像上随机矩形区域内的像素。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | p  | prob     | - |
|     | 参数2 | scale      | scale    | - |
|     | 参数3 | ratio      | ratio   | - |
|     | 参数4 | value      | value   | - |
|     | 参数5 | inplace      | inplace   | - |
|     | 参数6 | -     | max_attempts    | 生成随机擦除区域的最大尝试次数，超过该次数时将返回原始图像。 |

## 代码实例

```python
from download import download
from PIL import Image

url = "https://obs.dualstack.cn-north-4.myhuaweicloud.com/mindspore-website/notebook/datasets/flamingos.jpg"
download(url, './flamingos.jpg', replace=True)
orig_img = Image.open('flamingos.jpg')

# PyTorch
import torchvision.transforms as T

to_tensor = T.ToTensor()
transform = T.RandomErasing(p=0.5, scale=(0.02, 0.33))
img_torch = T.Compose([to_tensor, transform])(orig_img)


# MindSpore
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms

to_tensor = vision.ToTensor()
transform = vision.RandomErasing(prob=0.5, scale=(0.02, 0.33))
img_ms = transforms.Compose([to_tensor, transform])(orig_img)
```