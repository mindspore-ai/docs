# Function differences with torchvision.datasets.CIFAR10

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/CIFAR10.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torchvision.datasets.CIFAR10

```python
class torchvision.datasets.CIFAR10(
    root: str,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False)
```

For more information, see [torchvision.datasets.CIFAR10](https://pytorch.org/vision/0.14/generated/torchvision.datasets.CIFAR10.html).

## mindspore.dataset.Cifar10Dataset

```python
class mindspore.dataset.Cifar10Dataset(
    dataset_dir,
    usage=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None)
```

For more information, see [mindspore.dataset.Cifar10Dataset](https://mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.Cifar10Dataset.html#mindspore.dataset.Cifar10Dataset).

## Differences

PyTorch: Read the CIFAR-10 dataset. API integrates the transformation operations for image and label.

MindSpore: Read the CIFAR-10 dataset. Download dataset from web is not supported. Transforms for image and label depends on `mindshare.dataset.map` operation.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | train      | -    | Usage of this dataset，supported by `usage` in MindSpore |
|     | Parameter3 | transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter4 | target_transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter5 | download    | -   | Not supported by MindSpore |
|     | Parameter6 | -    | usage | Usage of this dataset |
|     | Parameter7 | -    | num_samples | The number of images to be included in the dataset. |
|     | Parameter8 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter9 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter10 | -    | sampler  | Object used to choose samples from the dataset |
|     | Parameter11 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter12 | -    | shard_id | The shard ID within num_shards |
|     | Parameter13 | -    | cache | Use tensor caching service to speed up dataset processing |

## Code Example

```python
# Torch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.CIFAR10(root, train=True, transform=T.RandomCrop((28, 28)))
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

root = "/path/to/dataset_directory/"
ms_dataloader = ds.Cifar10Dataset(root, usage='train')
ms_dataloader = ms_dataloader.map(vision.RandomCrop((28, 28)), ["image"])
```
