# Differences with torchaudio.datasets.SPEECHCOMMANDS

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/note/api_mapping/pytorch_diff/SPEECHCOMMANDS.md)

## torchaudio.datasets.SPEECHCOMMANDS

```python
class torchaudio.datasets.SPEECHCOMMANDS(
    root: str,
    url: str = 'speech_commands_v0.02',
    folder_in_archive: str = 'SpeechCommands',
    download: bool = False,
    subset: str = None)
```

For more information, see [torchaudio.datasets.SPEECHCOMMANDS](https://pytorch.org/audio/0.8.0/datasets.html#speechcommands).

## mindspore.dataset.SpeechCommandsDataset

```python
class mindspore.dataset.SpeechCommandsDataset(
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

For more information, see [mindspore.dataset.SpeechCommandsDataset](https://mindspore.cn/docs/en/r2.6.0rc1/api_python/dataset/mindspore.dataset.SpeechCommandsDataset.html#mindspore.dataset.SpeechCommandsDataset).

## Differences

PyTorch: Read the SpeechCommands dataset.

MindSpore: Read the SpeechCommands dataset. Downloading dataset from web is not supported.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
|Parameter | Parameter1 | root    | dataset_dir    | - |
|     | Parameter2 | url      | -  |Not supported by MindSpore  |
|     | Parameter3 | folder_in_archive      | - |Not supported by MindSpore   |
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
dataset = datasets.SPEECHCOMMANDS(root, url='speech_commands_v0.02')
dataloader = DataLoader(dataset)

# MindSpore
import mindspore.dataset as ds

# Download SpeechCommands dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── cat
#           ├── b433eff_nohash_0.wav
#           ├── 5a33edf_nohash_1.wav
#           └──....
#      ├── dog
#           ├── b433w2w_nohash_0.wav
#           └──....
#      ├── four
#      └── ....
root = "/path/to/dataset_directory/"
ms_dataloader = ds.SpeechCommandsDataset(root, usage='all')
```
