# Differences with torchaudio.datasets.CMUARCTIC

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/CMUARCTIC.md)

## torchaudio.datasets.CMUARCTIC

```python
class torchaudio.datasets.CMUARCTIC(
    root: str,
    url: str = 'aew',
    folder_in_archive: str = 'ARCTIC',
    download: bool = False)
```

For more information, see [torchaudio.datasets.CMUARCTIC](https://pytorch.org/audio/0.8.0/datasets.html#cmuarctic).

## mindspore.dataset.CMUArcticDataset

```python
class mindspore.dataset.CMUArcticDataset(
    dataset_dir,
    name=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None)
```

For more information, see [mindspore.dataset.CMUArcticDataset](https://mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.CMUArcticDataset.html#mindspore.dataset.CMUArcticDataset).

## Differences

PyTorch: Read the CMU Arctic dataset.

MindSpore: Read the CMU Arctic dataset. Downloading dataset from web is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | url      | name    |- |
|     | Parameter3 | folder_in_archive      | -    |Not supported by MindSpore |
|     | Parameter4 | download    | -   | Not supported by MindSpore |
|     | Parameter5 | -    | num_samples |  The number of images to be included in the dataset |
|     | Parameter6 | -    | num_parallel_workers | Number of worker threads to read the data |
|     | Parameter7 | -    | shuffle  | Whether to perform shuffle on the dataset |
|     | Parameter8 | -    | sampler  | Object used to choose samples from the dataset |
|     | Parameter9 | -    | num_shards | Number of shards that the dataset will be divided into |
|     | Parameter10 | -    | shard_id | The shard ID within num_shards |
|     | Parameter11 | -    | cache | Use tensor caching service to speed up dataset processing |

## Code Example

```python
# PyTorch
import torchaudio.datasets as datasets
from torch.utils.data import DataLoader

root = "/path/to/dataset_directory/"
dataset = datasets.CMUARCTIC(root, url='aew')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download CMUArctic dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── cmu_us_aew_arctic
#      │    ├── wav
#      │    │    ├──arctic_a0001.wav
#      │    │    ├──arctic_a0002.wav
#      │    │    ├──...
#      │    ├── etc
#      │    │    └── txt.done.data
#      ├── cmu_us_ahw_arctic
#      │    ├── wav
#      │    │    ├──arctic_a0001.wav
#      │    │    ├──arctic_a0002.wav
#      │    │    ├──...
#      │    └── etc
#      │         └── txt.done.data
#      └──...
root = "/path/to/dataset_directory/"
ms_dataloader = ds.CMUArcticDataset(root, name='aew')
```
