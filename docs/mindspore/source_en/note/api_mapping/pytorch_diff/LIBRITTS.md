# Differences with torchaudio.datasets.LIBRITTS

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/LIBRITTS.md)

## torchaudio.datasets.LIBRITTS

```python
class torchaudio.datasets.LIBRITTS(
    root: str,
    url: str = 'train-clean-100',
    folder_in_archive: str = 'LibriTTS',
    download: bool = False)
```

For more information, see [torchaudio.datasets.LIBRITTS](https://pytorch.org/audio/0.8.0/datasets.html#libritts).

## mindspore.dataset.LibriTTSDataset

```python
class mindspore.dataset.LibriTTSDataset(
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

For more information, see [mindspore.dataset.LibriTTSDataset](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.LibriTTSDataset.html#mindspore.dataset.LibriTTSDataset).

## Differences

PyTorch: Read the LibriTTS dataset.

MindSpore: Read the LibriTTS dataset. Downloading dataset from web is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | url      | usage    |- |
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
dataset = datasets.LIBRITTS(root, url='train-clean-100')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download LibriTTS dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── dev-clean
#      │    ├── 116
#      │    │    ├── 288045
#      |    |    |    ├── 116_288045.trans.tsv
#      │    │    │    ├── 116_288045_000003_000000.wav
#      │    │    │    └──...
#      │    │    ├── 288046
#      |    |    |    ├── 116_288046.trans.tsv
#      |    |    |    ├── 116_288046_000003_000000.wav
#      │    |    |    └── ...
#      |    |    └── ...
#      │    ├── 1255
#      │    │    ├── 138279
#      |    |    |    ├── 1255_138279.trans.tsv
#      │    │    │    ├── 1255_138279_000001_000000.wav
#      │    │    │    └── ...
#      │    │    ├── 74899
#      |    |    |    ├── 1255_74899.trans.tsv
#      |    |    |    ├── 1255_74899_000001_000000.wav
#      │    |    |    └── ...
#      |    |    └── ...
#      |    └── ...
#      └── ...
root = "/path/to/dataset_directory/"
ms_dataloader = ds.LibriTTSDataset(root, usage='train-clean-100')
```
