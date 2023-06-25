# 比较与torchvision.transforms.RandomResizedCrop的差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomResizedCrop.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchvision.transforms.RandomResizedCrop

```python
class torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=InterpolationMode.BILINEAR, antialias: Optional[bool] = None)
```

更多内容详见[torchvision.transforms.RandomResizedCrop](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomResizedCrop)。

## mindspore.dataset.vision.RandomResizedCrop

```python
class mindspore.dataset.vision.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Inter.BILINEAR, max_attempts=10)
```

更多内容详见[mindspore.dataset.vision.RandomResizedCrop](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_vision/mindspore.dataset.vision.RandomResizedCrop.html)。

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
