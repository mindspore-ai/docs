# 比较与torchaudio.datasets.SPEECHCOMMANDS的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/SPEECHCOMMANDS.md)

## torchaudio.datasets.SPEECHCOMMANDS

```python
class torchaudio.datasets.SPEECHCOMMANDS(
    root: str,
    url: str = 'speech_commands_v0.02',
    folder_in_archive: str = 'SpeechCommands',
    download: bool = False,
    subset: str = None)
```

更多内容详见[torchaudio.datasets.SPEECHCOMMANDS](https://pytorch.org/audio/0.8.0/datasets.html#speechcommands)。

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

更多内容详见[mindspore.dataset.SpeechCommandsDataset](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.SpeechCommandsDataset.html#mindspore.dataset.SpeechCommandsDataset)。

## 差异对比

PyTorch：读取SpeechCommands数据集。

MindSpore：读取SpeechCommands数据集，不支持下载。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | url      | -    |MindSpore不支持|
|     | 参数3 | folder_in_archive      | -    | MindSpore不支持|
|     | 参数4 | download    | -   | MindSpore不支持 |
|     | 参数5 | subset      | usage    | - |
|     | 参数6 | -    | num_samples | 指定从数据集中读取的样本数 |
|     | 参数7 | -    | num_parallel_workers | 指定读取数据的工作线程数 |
|     | 参数8 | -    | shuffle  | 指定是否混洗数据集 |
|     | 参数9 | -    | sampler  | 指定采样器 |
|     | 参数10 | -    | num_shards | 指定分布式训练时将数据集进行划分的分片数 |
|     | 参数11 | -    | shard_id | 指定分布式训练时使用的分片ID号 |
|     | 参数12 | -    | cache | 指定单节点数据缓存服务 |

## 代码示例

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
