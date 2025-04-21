# Differences with torchvision.datasets.CelebA

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/CelebA.md)

## torchvision.datasets.CelebA

```python
class torchvision.datasets.CelebA(
    root: str,
    split: str = 'train',
    target_type: Union[List[str], str] = 'attr',
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False)
```

For more information, see [torchvision.datasets.CelebA](https://pytorch.org/vision/0.9/datasets.html#torchvision.datasets.CelebA).

## mindspore.dataset.CelebADataset

```python
class mindspore.dataset.CelebADataset(
    dataset_dir,
    num_parallel_workers=None,
    shuffle=None,
    usage='all',
    sampler=None,
    decode=False,
    extensions=None,
    num_samples=None,
    num_shards=None,
    shard_id=None,
    cache=None,
    decrypt=None)
```

For more information, see [mindspore.dataset.CelebADataset](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.CelebADataset.html#mindspore.dataset.CelebADataset).

## Differences

PyTorch: Read the CelebA (CelebFaces Attributes) dataset. API integrates the transformation operations for image and label.

MindSpore: Read the CelebA (CelebFaces Attributes) dataset. Downloading dataset from web is not supported. Transforms for image and label depends on `mindshare.dataset.map` operation.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | split      | usage    |- |
|     | Parameter3 | target_type      | -    | - |
|     | Parameter4 | transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter5 | target_transform    | -   | Supported by `mindspore.dataset.map` operation |
|     | Parameter6 | download    | -   | Not supported by MindSpore |
|     | Parameter7 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter8 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter9 | -    | sampler  | Object used to choose samples from the dataset |
|     | Parameter10 | -    | decode | Whether to decode the images after reading |
|     | Parameter11 | -    | extensions | List of file extensions to be included in the dataset |
|     | Parameter12 | -    | num_samples | The number of images to be included in the dataset |
|     | Parameter13 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter14 | -    | shard_id | The shard ID within num_shards |
|     | Parameter15 | -    | cache | Use tensor caching service to speed up dataset processing |
|     | Parameter16 | -    | decrypt | Image decryption function |

## Code Example

```python
# PyTorch
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.CelebA(root, split='train', target_type="attr", transform=T.ToTensor(), download=True)
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

# Download CelebA dataset files, unzip the img_align_celeba.zip and put list_attr_celeba.txt together like
# .
# └── /path/to/dataset_directory/
#      ├── list_attr_celeba.txt
#      ├── 000001.jpg
#      ├── 000002.jpg
#      ├── 000003.jpg
#      ├── ...
root = "/path/to/dataset_directory/"
ms_dataloader = ds.CelebADataset(root, usage='train', decode=True)
ms_dataloader = ms_dataloader.map(vision.ToTensor(), ["image"])
```
