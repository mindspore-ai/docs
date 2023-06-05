# 比较与torchvision.transforms.RandomAutocontrast的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomAutocontrast.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchvision.transforms.RandomAutocontrast

```python
class torchvision.transforms.RandomAutocontrast(p=0.5)
```

更多内容详见[torchvision.transforms.RandomAutocontrast](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomAutocontrast.html)。

## mindspore.dataset.vision.RandomAutocontrast

```python
class mindspore.dataset.vision.RandomAutoContrast(cutoff=0.0, ignore=None, prob=0.5)
```

更多内容详见[mindspore.dataset.vision.RandomAutocontrast](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.RandomAutocontrast.html)。

## 差异对比

PyTorch：对张量图像以给定的概率自动调整图像的对比度。

MindSpore：对输入图像以给定的概率自动调整图像的对比度。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | p    | prob    | - |
|     | 参数2 | -     | cutoff    | 图像直方图中最亮和最暗的像素范围 |
|     | 参数3 | -     | ignore     | 要忽略的像素值，比如背景像素 |
