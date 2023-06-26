# Differences with torchvision.transforms.RandomResizedCrop

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomResizedCrop.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.RandomResizedCrop

```python
class torchvision.transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=InterpolationMode.BILINEAR)
```

For more information, see [torchvision.transforms.RandomResizedCrop](https://pytorch.org/vision/0.9/transforms.html#torchvision.transforms.RandomResizedCrop).

## mindspore.dataset.vision.RandomResizedCrop

```python
class mindspore.dataset.vision.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Inter.BILINEAR, max_attempts=10)
```

For more information, see [mindspore.dataset.vision.RandomResizedCrop](https://mindspore.cn/docs/en/master/api_python/dataset_vision/mindspore.dataset.vision.RandomResizedCrop.html).

## Differences

PyTorch: Crop a random portion of image and resize it to a given size.

MindSpore: Crop a random portion of image and resize it to a given size.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | size    | size    | - |
|     | Parameter2 | scale      | scale   |- |
|     | Parameter3 | ratio     | ratio    | - |
|     | Parameter4 | interpolation     | interpolation   | - |
|     | Parameter5 | -     | max_attempts   | The maximum number of attempts to propose a valid crop_area |