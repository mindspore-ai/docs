# Differences with torchvision.datasets.Cityscapes

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/Cityscapes.md)

## torchvision.datasets.Cityscapes

```python
class torchvision.datasets.Cityscapes(
    root: str,
    split: str,
    mode: str,
    target_type: str or list,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    transforms: Optional[Callable] = None
    )
```

For more information, see [torchvision.datasets.Cityscapes](https://pytorch.org/vision/0.9/datasets.html#cityscapes).

## mindspore.dataset.CityscapesDataset

```python
class mindspore.dataset.CityscapesDataset(
    dataset_dir,
    usage='train',
    quality_mode='fine',
    task='instance',
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    decode=False,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None
    )
```

For more information, see [mindspore.dataset.CityscapesDataset](https://www.mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.CityscapesDataset.html).

## Differences

PyTorch: Read the Cityscapes dataset.

MindSpore: Read the Cityscapes dataset. Downloading dataset from web is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | split      | usage    | - |
|     | Parameter3 | mode    | quality_mode   | - |
|     | Parameter4 | target_type    | task   | - |
|     | Parameter5 | transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter6 | target_transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter7 | transforms    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter8 | -    | num_samples | The number of images to be included in the dataset. |
|     | Parameter9 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter10 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter11 | -    | decode  | Decode the images after reading |
|     | Parameter12 | -    | sampler  | Object used to choose samples from the dataset |
|     | Parameter13 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter14 | -    | shard_id | The shard ID within num_shards |
|     | Parameter15 | -    | cache | Use tensor caching service to speed up dataset processing |

## Code Example

```python
# PyTorch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.Cityscapes(root, split='train', mode='fine', target_type='semantic')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

# Download the dataset files, unzip into the following structure
# .
# └── "/path/to/dataset_directory"
#      ├── leftImg8bit
#      |    ├── train
#      |    |    ├── aachen
#      |    |    |    ├── aachen_000000_000019_leftImg8bit.png
#      |    |    |    ├── aachen_000001_000019_leftImg8bit.png
#      |    |    |    ├── ...
#      |    |    ├── bochum
#      |    |    |    ├── ...
#      |    |    ├── ...
#      |    ├── test
#      |    |    ├── ...
#      |    ├── val
#      |    |    ├── ...
#      └── gtFine
#           ├── train
#           |    ├── aachen
#           |    |    ├── aachen_000000_000019_gtFine_color.png
#           |    |    ├── aachen_000000_000019_gtFine_instanceIds.png
#           |    |    ├── aachen_000000_000019_gtFine_labelIds.png
#           |    |    ├── aachen_000000_000019_gtFine_polygons.json
#           |    |    ├── aachen_000001_000019_gtFine_color.png
#           |    |    ├── aachen_000001_000019_gtFine_instanceIds.png
#           |    |    ├── aachen_000001_000019_gtFine_labelIds.png
#           |    |    ├── aachen_000001_000019_gtFine_polygons.json
#           |    |    ├── ...
#           |    ├── bochum
#           |    |    ├── ...
#           |    ├── ...
#           ├── test
#           |    ├── ...
#           └── val
#                ├── ...

root = "/path/to/dataset_directory/"
ms_dataloader = ds.CityscapesDataset(root, usage='train')
ms_dataloader = ms_dataloader.map(vision.RandomCrop((28, 28)), ["image"])
```
