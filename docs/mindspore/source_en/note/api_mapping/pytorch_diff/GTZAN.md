# Differences with torchaudio.datasets.GTZAN

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/GTZAN.md)

## torchaudio.datasets.GTZAN

```python
class torchaudio.datasets.GTZAN(
    root: str,
    url: str = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz',
    folder_in_archive: str = 'genres',
    download: bool = False,
    subset: str = None)
```

For more information, see [torchaudio.datasets.GTZAN](https://pytorch.org/audio/0.8.0/datasets.html#gtzan).

## mindspore.dataset.GTZANDataset

```python
class mindspore.dataset.GTZANDataset(
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

For more information, see [mindspore.dataset.GTZANDataset](https://mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.GTZANDataset.html#mindspore.dataset.GTZANDataset).

## Differences

PyTorch: Read the GTZAN dataset.

MindSpore: Read the GTZAN dataset. Downloading dataset from web is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | url      |  -  |Not supported by MindSpore |
|     | Parameter3 | folder_in_archive      | -  |Not supported by MindSpore   |
|     | Parameter4 | download    | -   | Not supported by MindSpore |
|     | Parameter5 | subset      | usage    |- |
|     | Parameter6 | -    | num_samples |  The number of images to be included in the dataset |
|     | Parameter7 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter8 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter9 | -    | sampler  | Object used to choose samples from the dataset |
|     | Parameter10 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter11 | -    | shard_id | The shard ID within num_shards |
|     | Parameter12 | -    | cache | Use tensor caching service to speed up dataset processing |

## Code Example

```python
# PyTorch
import torchaudio.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.GTZAN(root, url='http://opihi.cs.uvic.ca/sound/genres.tar.gz')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download GTZAN dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── blues
#      │    ├──blues.00000.wav
#      │    ├──blues.00001.wav
#      │    ├──blues.00002.wav
#      │    ├──...
#      ├── disco
#      │    ├──disco.00000.wav
#      │    ├──disco.00001.wav
#      │    ├──disco.00002.wav
#      │    └──...
#      └──...
root = "/path/to/dataset_directory/"
ms_dataloader = ds.GTZANDataset(root, usage='all')
```
