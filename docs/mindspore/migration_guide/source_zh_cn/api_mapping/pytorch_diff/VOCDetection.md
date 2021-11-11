# 比较与torchvision.datasets.VOCDetection的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/VOCDetection.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torchvision.datasets.VOCDetection

```python
class torchvision.datasets.VOCDetection(
    root: str,
    year: str='2012',
    image_set: str='train',
    download: bool=False,
    transform: Optional[Callable]=None,
    target_transform: Optional[Callable]=None,
    transforms: Optional[Callable]=None
    )
```

更多内容详见[torchvision.datasets.VOCDetection](https://pytorch.org/vision/0.10/datasets.html#torchvision.datasets.VOCDetection)。

## mindspore.dataset.VOCDataset

```python
class mindspore.dataset.VOCDataset(
    dataset_dir,
    task="Segmentation",
    usage="train",
    class_indexing=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    decode=False,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None,
    extra_metadata=False
    )
```

更多内容详见[mindspore.dataset.VOCDataset](https://mindspore.cn/docs/api/zh-CN/r1.5/api_python/dataset/mindspore.dataset.VOCDataset.html#mindspore.dataset.VOCDataset)。

## 使用方式

PyTorch：生成PASCAL VOC 目标检测格式数据集。

MindSpore：用于读取和分析VOC数据集的源数据集。

## 代码示例

```python
import mindspore.dataset as ds
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# In MindSpore, the generated dataset with different task setting has different output columns.

voc_dataset_dir = "/path/to/voc_dataset_directory/"

# task = Detection, output columns: [image, dtype=uint8], [bbox, dtype=float32], [label, dtype=uint32], [difficult, dtype=uint32], [truncate, dtype=uint32].
dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir, task="Detection", usage="train")
for item in dataset:
    print("item:", item[0])
    print(len(item[0]))
    break
# out:
# item: [255 216 255 ...   3 255 217]
# 147025

# In torch, the output will be result of transform, eg. RandomCrop
root = "/path/to/voc_dataset_directory/"

dataset = datasets.VOCDetection(root, image_set='train', year='2012', transform=T.ToTensor())
dataloader = DataLoader(dataset=dataset, num_workers=8, batch_size=1, shuffle=True)
for epoch in range(1):
    for i, (data, label) in enumerate(dataloader):
        print((data, label)[0])

# out:
# tensor([[[[0.7176, 0.7176, 0.7216,  ..., 0.7843, 0.7843, 0.7843],
#           [0.7216, 0.7216, 0.7216,  ..., 0.7882, 0.7882, 0.7882],
#           [0.7216, 0.7255, 0.7255,  ..., 0.7882, 0.7882, 0.7882],
#           ...,
#           ...
#          ...,
#           [0.6667, 0.6667, 0.6667,  ..., 0.8118, 0.8118, 0.8078],
#           [0.6627, 0.6627, 0.6588,  ..., 0.8078, 0.8039, 0.8000],
#           [0.6627, 0.6627, 0.6588,  ..., 0.8078, 0.8039, 0.8000]]]])
#  {'annotation': {'folder': ['VOC2012'], 'filename': ['61.jpg'], 'source': {'database': ['simulate VOC2007 Database'],
#  'annotation': ['simulate VOC2007'], 'image': ['flickr']}, 'size': {'width': ['500'], 'height': ['333'], 'depth': ['3']}, 'segmented': ['1'],
#  'object': [{'name': ['train'], 'pose': ['Unspecified'], 'truncated': ['0'], 'difficult': ['0'], 'bndbox': {'xmin': ['252'], 'ymin': ['42'],
#  'xmax': ['445'], 'ymax': ['282']}}, {'name': ['person'], 'pose': ['Frontal'], 'truncated': ['0'], 'difficult': ['0'], 'bndbox': {'xmin': ['204'],
#  'ymin': ['198'], 'xmax': ['271'], 'ymax': ['293']}}]}}
```
