# Differences with torchaudio.datasets.YESNO

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/YESNO.md)

## torchaudio.datasets.YESNO

```python
class torchaudio.datasets.YESNO(
    root: str,
    url: str = 'http://www.openslr.org/resources/1/waves_yesno.tar.gz',
    folder_in_archive: str = 'waves_yesno',
    download: bool = False)
```

For more information, see [torchaudio.datasets.YESNO](https://pytorch.org/audio/0.8.0/datasets.html#yesno).

## mindspore.dataset.YesNoDataset

```python
class mindspore.dataset.YesNoDataset(
    dataset_dir,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None)
```

For more information, see [mindspore.dataset.YesNoDataset](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.YesNoDataset.html#mindspore.dataset.YesNoDataset).

## Differences

PyTorch: Read the YesNo dataset.

MindSpore: Read the YesNo dataset. Downloading dataset from web is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | url      | -    |Not supported by MindSpore|
|     | Parameter3 | folder_in_archive      | - |Not supported by MindSpore|
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
dataset = datasets.YESNO(root, url='http://www.openslr.org/resources/1/waves_yesno.tar.gz')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download YesNo dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── 1_1_0_0_1_1_0_0.wav
#      ├── 1_0_0_0_1_1_0_0.wav
#      ├── 1_1_0_0_1_1_0_0.wav
#      └──....
root = "/path/to/dataset_directory/"
ms_dataloader = ds.YesNoDataset(root)
```
