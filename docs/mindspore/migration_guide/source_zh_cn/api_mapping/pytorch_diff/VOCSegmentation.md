# 比较与torchvision.datasets.VOCSegmentation的功能差异

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_zh_cn/api_mapping/pytorch_diff/VOCSegmentation.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>

## torchvision.datasets.VOCSegmentation

```python
class torchvision.datasets.VOCSegmentation(
    root: str,
    year: str='2012',
    image_set: str='train',
    download: bool=False,
    transform: Optional[Callable]=None,
    target_transform: Optional[Callable]=None,
    transforms: Optional[Callable]=None
    )
```

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

## 使用方式

PyTorch：生成PASCAL VOC图像分割格式数据集。

MindSpore：用于读取和分析VOC数据集的源数据集。

## 代码示例

```python
import mindspore.dataset as ds
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# In MindSpore, the generated dataset with different task setting has different output columns.

voc_dataset_dir = "/path/to/voc_dataset_directory/"

# task = Segmentation, output columns: [image, dtype=uint8], [target,dtype=uint8].
dataset = ds.VOCDataset(dataset_dir=voc_dataset_dir,
                                                    task="Segmentation",
                                                    usage="train")
for item in dataset:
    print("item:", item[0])
    print(len(item[0]))
    break
# Out:
# item: [255 216 255 ...  73 255 217]
# 52544

# In torch, the output will be result of transform, eg. RandomCrop
root = "/path/to/voc_dataset_directory/"

dataset = datasets.Segmentation(root, image_set='train', year='2012', transform=T.RandomCrop(300))
print(dataset)
print(type(dataset))
```
