# Differences with torchaudio.datasets.LJSPEECH

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/LJSPEECH.md)

## torchaudio.datasets.LJSPEECH

```python
class torchaudio.datasets.LJSPEECH(
    root: str,
    url: str = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2',
    folder_in_archive: str = 'wavs',
    download: bool = False)
```

For more information, see [torchaudio.datasets.LJSPEECH](https://pytorch.org/audio/0.8.0/datasets.html#ljspeech).

## mindspore.dataset.LJSpeechDataset

```python
class mindspore.dataset.LJSpeechDataset(
    dataset_dir,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=None,
    sampler=None,
    num_shards=None,
    shard_id=None,
    cache=None)
```

For more information, see [mindspore.dataset.LJSpeechDataset](https://mindspore.cn/docs/en/br_base/api_python/dataset/mindspore.dataset.LJSpeechDataset.html#mindspore.dataset.LJSpeechDataset).

## Differences

PyTorch: Read the LJSpeech dataset.

MindSpore: Read the LJSpeech dataset. Downloading dataset from web is not supported.

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
dataset = datasets.LJSPEECH(root, url='https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download LJSpeech dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── README
#      ├── metadata.csv
#      └── wavs
#          ├── LJ001-0001.wav
#          ├── LJ001-0002.wav
#          ├── LJ001-0003.wav
#          ├── LJ001-0004.wav
#          ├── LJ001-0005.wav
#          ├── LJ001-0006.wav
#          ├── LJ001-0007.wav
#          ├── LJ001-0008.wav
#           ...
#          ├── LJ050-0277.wav
#          └── LJ050-0278.wav
root = "/path/to/dataset_directory/"
ms_dataloader = ds.LJSpeechDataset(root)
```
