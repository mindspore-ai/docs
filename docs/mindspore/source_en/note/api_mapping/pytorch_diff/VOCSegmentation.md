# Differences with torchvision.datasets.VOCSegmentation

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/VOCSegmentation.md)

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

For more information, see [torchvision.datasets.VOCSegmentation](https://pytorch.org/vision/0.9/datasets.html#torchvision.datasets.VOCSegmentation).

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

PyTorch: Pascal VOC Segmentation Dataset.

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
|     | Parameter18 | -    | extra_metadata | Specifies whether to output an additional column of data representing image meta-information.|
|     | Parameter19 | -    | decrypt | Image decryption function |

## Code Example

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
root = "/path/to/voc_dataset_directory2/"

dataset = datasets.VOCSegmentation(root, image_set='train', year='2012', transform=T.RandomCrop(300))
print(dataset)
print(type(dataset))
```
