# 比较与torchtext.datasets.IWSLT2017的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/IWSLT2017.md)

## torchtext.datasets.IWSLT2017

```python
class torchtext.datasets.IWSLT2017(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'valid', 'test'),
    language_pair: Sequence = ('de', 'en'))
```

更多内容详见[torchtext.datasets.IWSLT2017](https://pytorch.org/text/0.9.0/datasets.html#iwslt2017)。

## mindspore.dataset.IWSLT2017Dataset

```python
class mindspore.dataset.IWSLT2017Dataset(
    dataset_dir,
    usage=None,
    language_pair=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None)
```

更多内容详见[mindspore.dataset.IWSLT2017Dataset](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset/mindspore.dataset.IWSLT2017Dataset.html#mindspore.dataset.IWSLT2017Dataset)。

## 差异对比

PyTorch：读取IWSLT2017数据集。

MindSpore：读取IWSLT2017数据集，不支持下载。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | split      | usage    |- |
|     | 参数3 | language_pair      | language_pair    |- |
|     | 参数4 | -    | num_samples | 指定从数据集中读取的样本数 |
|     | 参数5 | -    | num_parallel_workers | 指定读取数据的工作线程数 |
|     | 参数6 | -    | shuffle  | 指定是否混洗数据集 |
|     | 参数7 | -    | num_shards | 指定分布式训练时将数据集进行划分的分片数 |
|     | 参数8 | -    | shard_id | 指定分布式训练时使用的分片ID号 |
|     | 参数9 | -    | cache | 指定单节点数据缓存服务 |

## 代码示例

```python
# PyTorch
import torchtext.datasets as datasets

root = "/path/to/dataset_root/"
train_iter, valid_iter, test_iter = datasets.IWSLT2017(root, split=('train', 'valid', 'test'))
data = next(iter(train_iter))

# MindSpore
import mindspore.dataset as ds

# Download IWSLT2017 dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      └── DeEnItNlRo
#          └── DeEnItNlRo
#              └── DeEnItNlRo-DeEnItNlRo
#                  ├── IWSLT17.TED.dev2010.de-en.de.xml
#                  ├── train.tags.de-en.de
#                  ├── ...
root = "/path/to/dataset_directory/"
dataset = ds.IWSLT2017Dataset(root, usage='all')
data = next(iter(dataset))
```
