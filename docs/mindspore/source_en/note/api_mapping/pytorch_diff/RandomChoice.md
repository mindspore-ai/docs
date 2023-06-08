# Function Differences with torchvision.transforms.RandomChoice

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/RandomChoice.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.transforms.RandomChoice

```python
class torchvision.transforms.RandomChoice(transforms, p=None)
```

For more information, see [torchvision.transforms.RandomChoice](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomChoice.html).

## mindspore.dataset.transforms.RandomChoice

```python
class mindspore.dataset.transforms.RandomChoice(transforms)
```

For more information, see [mindspore.dataset.transforms.RandomChoice](https://mindspore.cn/docs/en/master/api_python/dataset_transforms/mindspore.dataset.transforms.RandomChoice.html).

## Differences

PyTorch: Apply single transformation randomly picked from a list. The probability can be specified.

MindSpore: Apply single transformation randomly picked from a list. The probability can not be specified manually.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | transforms  | transforms    | - |
|     | Parameter2 | p     | -   | Specifies the probability of selecting a enhancement |