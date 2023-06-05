# Function Differences with torchvision.transforms.RandomAutocontrast

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomAutocontrast.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.RandomAutocontrast

```python
class torchvision.transforms.RandomAutocontrast(p=0.5)
```

For more information, see [torchvision.transforms.RandomAutocontrast](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomAutocontrast.html).

## mindspore.dataset.vision.RandomAutocontrast

```python
class mindspore.dataset.vision.RandomAutoContrast(cutoff=0.0, ignore=None, prob=0.5)
```

For more information, see [mindspore.dataset.vision.RandomAutocontrast](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.RandomAutoContrast.html).

## Differences

PyTorch: Automatically adjust the contrast of a tensor image with a given probability.

MindSpore: Automatically adjust the contrast of the input image with a given probability.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | p    | prob    | - |
|     | Parameter2 | -     | cutoff    | Range of the brightest and darkest pixels in the image histogram |
|     | Parameter3 | -     | ignore     | Value of the pixel to be ignored, for example, the background pixel |