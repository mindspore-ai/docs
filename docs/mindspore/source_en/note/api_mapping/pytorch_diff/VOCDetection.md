# Differences with torchvision.datasets.VOCDetection

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/VOCDetection.md)

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

For more information, see [torchvision.datasets.VOCDetection](https://pytorch.org/vision/0.9/datasets.html#torchvision.datasets.VOCDetection).

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
    extra_metadata=False,
    decrypt=None
    )
```

For more information, see [mindspore.dataset.VOCDataset](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.VOCDataset.html#mindspore.dataset.VOCDataset).

## Differences

PyTorch: Pascal VOC Detection Dataset.

MindSpore: A source dataset for reading and parsing VOC dataset.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | year      | -    | Not supported by MindSpore |
|     | Parameter3 | image_set      | usage  |- |
|     | Parameter4 | download      | -    | Not supported by MindSpore |
|     | Parameter5 | transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter6 | target_transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter7 | transforms    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter8 | -      | task  | Set the task type of reading voc data |
|     | Parameter9 | -      | class_indexing  | A str-to-int mapping from label name to index |
|     | Parameter10 | -    | num_samples |  The number of images to be included in the dataset |
|     | Parameter11 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter12 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter13 | -   | decode | Whether to decode the images after reading |
|     | Parameter14 | -    | sampler  | Object used to choose samples from the dataset |
|     | Parameter15 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter16 | -    | shard_id | The shard ID within num_shards |
|     | Parameter17 | -    | cache | Use tensor caching service to speed up dataset processing |
|     | Parameter18 | -    | extra_metadata | Flag to add extra meta-data to row |
|     | Parameter19 | -    | decrypt | Image decryption function |

## Code Example

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
# Out:
# item: [255 216 255 ...   3 255 217]
# 147025

# In torch, the output will be result of transform, eg. RandomCrop
root = "/path/to/voc_dataset_directory2/"

dataset = datasets.VOCDetection(root, image_set='train', year='2012', transform=T.ToTensor())
dataloader = DataLoader(dataset=dataset, num_workers=8, batch_size=1, shuffle=True)
for epoch in range(1):
    for i, (data, label) in enumerate(dataloader):
        print((data, label)[0])

# Out:
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
