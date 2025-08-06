# 比较与torchaudio.datasets.CMUARCTIC的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/CMUARCTIC.md)

## torchaudio.datasets.CMUARCTIC

```python
class torchaudio.datasets.CMUARCTIC(
    root: str,
    url: str = 'aew',
    folder_in_archive: str = 'ARCTIC',
    download: bool = False)
```

更多内容详见[torchaudio.datasets.CMUARCTIC](https://pytorch.org/audio/0.8.0/datasets.html#cmuarctic)。

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

更多内容详见[mindspore.dataset.CMUArcticDataset](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset/mindspore.dataset.CMUArcticDataset.html#mindspore.dataset.CMUArcticDataset)。

## 差异对比

PyTorch：读取CMU Arctic数据集。

MindSpore：读取CMU Arctic数据集，不支持下载。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | url      | name    |- |
|     | 参数3 | folder_in_archive      | -    | MindSpore不支持|
|     | 参数4 | download    | -   | MindSpore不支持 |
|     | 参数5 | -    | num_samples | 指定从数据集中读取的样本数 |
|     | 参数6 | -    | num_parallel_workers | 指定读取数据的工作线程数 |
|     | 参数7 | -    | shuffle  | 指定是否混洗数据集 |
|     | 参数8 | -    | sampler  | 指定采样器 |
|     | 参数9 | -    | num_shards | 指定分布式训练时，将数据集进行划分的分片数 |
|     | 参数10 | -    | shard_id | 指定分布式训练时，使用的分片ID号 |
|     | 参数11 | -    | cache | 指定单节点数据缓存服务 |

## 代码示例

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
