# Differences with torchvision.datasets.ImageFolder

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ImageFolder.md)

## torchvision.datasets.ImageFolder

```python
class torchvision.datasets.ImageFolder(
    root: str,
    transform: Optional[Callable] = None,
    target_transform: Union[Callable, NoneType] = None,
    loader: Optional[Callable] = None,
    is_valid_file: bool = None)
```

For more information, see [torchvision.datasets.ImageFolder](https://pytorch.org/vision/0.9/datasets.html#torchvision.datasets.ImageFolder).

## mindspore.dataset.ImageFolderDataset

```python
class mindspore.dataset.ImageFolderDataset(
    dataset_dir,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    sampler=None,
    extensions=None,
    class_indexing=None,
    decode=False,
    num_shards=None,
    shard_id=None,
    cache=None,
    decrypt=None)
```

For more information, see [mindspore.dataset.ImageFolderDataset](https://mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.ImageFolderDataset.html#mindspore.dataset.ImageFolderDataset).

## Differences

PyTorch: A source dataset that reads images from a tree of directories. API integrates the transformation operations for image and label. File Loader can be specified.

MindSpore: A source dataset that reads images from a tree of directories. Transforms for image and label depends on `mindshare.dataset.map` operation. File Loader can not be specified.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter3 | target_transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter4 | loader    | -   | Not supported by MindSpore |
|     | Parameter5 | is_valid_file    | -   | Not supported by MindSpore |
|     | Parameter6 | -    | num_samples | The number of images to be included in the dataset |
|     | Parameter7 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter8 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter9 | -    | sampler  | Object used to choose samples from the dataset |
|     | Parameter10 | -    | extensions | List of file extensions to be included in the dataset |
|     | Parameter11 | -    | class_indexing | A str-to-int mapping from folder name to index |
|     | Parameter12 | -    | decode | Whether to decode the images after reading |
|     | Parameter13 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter14 | -    | shard_id | The shard ID within num_shards |
|     | Parameter15 | -    | cache | Use tensor caching service to speed up dataset processing |
|     | Parameter16 | -    | decrypt | Image decryption function |

## Code Example

Assume that we have a directory with the following structure:

```text
imageset/
    ├── cat
    │   ├── cat_0.jpg
    │   ├── cat_1.jpg
    │   └── cat_2.jpg
    ├── fish
    │   ├── fish_0.jpg
    │   ├── fish_1.jpg
    │   ├── fish_2.jpg
    │   └── fish_3.jpg
    ├── fruits
    │   ├── fruits_0.jpg
    │   ├── fruits_1.jpg
    │   └── fruits_2.jpg
    ├── plane
    │   ├── plane_0.jpg
    │   ├── plane_1.jpg
    │   └── plane_2.jpg
    └── tree
        ├── tree_0.jpg
        ├── tree_1.jpg
        └── tree_2.jpg
```

```python
# Torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/imageset/"
dataset = datasets.ImageFolder(root, transform=T.RandomCrop((256, 256)))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

root = "/path/to/imageset/"
ms_dataloader = ds.ImageFolderDataset(root, decode=True)
ms_dataloader = ms_dataloader.map(vision.RandomCrop((256, 256)), ["image"])
```
