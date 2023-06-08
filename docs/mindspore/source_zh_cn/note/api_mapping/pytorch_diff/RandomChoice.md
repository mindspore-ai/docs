# 比较与torchvision.transforms.RandomChoice的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/RandomChoice.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## torchvision.transforms.RandomChoice

```python
class torchvision.transforms.RandomChoice(transforms, p=None)
```

更多内容详见[torchvision.transforms.RandomChoice](https://pytorch.org/vision/0.14/generated/torchvision.transforms.RandomChoice.html)。

## mindspore.dataset.transforms.RandomChoice

```python
class mindspore.dataset.transforms.RandomChoice(transforms)
```

更多内容详见[mindspore.dataset.transforms.RandomChoice](https://mindspore.cn/docs/zh-CN/master/api_python/dataset_transforms/mindspore.dataset.transforms.RandomChoice.html)。

## 差异对比

PyTorch：在一组数据增强中随机选择部分增强处理进行应用，支持指定选择概率。

MindSpore：在一组数据增强中随机选择部分增强处理进行应用，不支持指定选择概率。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | transforms  | transforms    | - |
|     | 参数2 | p     | -   | 指定选择增强处理的概率 |
