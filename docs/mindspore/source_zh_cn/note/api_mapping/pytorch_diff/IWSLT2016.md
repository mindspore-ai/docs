# 比较与torchtext.datasets.IWSLT2016的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/IWSLT2016.md)

## torchtext.datasets.IWSLT2016

```python
class torchtext.datasets.IWSLT2016(
    root: str = '.data',
    split: Union[List[str], str] = ('train', 'valid', 'test'),
    language_pair: Sequence =('de', 'en'),
    valid_set: str ='tst2013',
    test_set: str ='tst2014')
```

更多内容详见[torchtext.datasets.IWSLT2016](https://pytorch.org/text/0.9.0/datasets.html#iwslt2016)。

## mindspore.dataset.IWSLT2016Dataset

```python
class mindspore.dataset.IWSLT2016Dataset(
    dataset_dir,
    usage=None,
    language_pair=None,
    valid_set=None,
    test_set=None,
    num_samples=None,
    num_parallel_workers=None,
    shuffle=Shuffle.GLOBAL,
    num_shards=None,
    shard_id=None,
    cache=None)
```

更多内容详见[mindspore.dataset.IWSLT2016Dataset](https://mindspore.cn/docs/zh-CN/br_base/api_python/dataset/mindspore.dataset.IWSLT2016Dataset.html#mindspore.dataset.IWSLT2016Dataset)。

## 差异对比

PyTorch：读取IWSLT2016数据集。

MindSpore：读取IWSLT2016数据集，不支持下载。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | root    | dataset_dir    | - |
|     | 参数2 | split      | usage    |- |
|     | 参数3 | language_pair      | language_pair    |- |
|     | 参数4 | valid_set      | valid_set    |- |
|     | 参数5 | test_set      | test_set    |- |
|     | 参数6 | -    | num_samples | 指定从数据集中读取的样本数 |
|     | 参数7 | -    | num_parallel_workers | 指定读取数据的工作线程数 |
|     | 参数8 | -    | shuffle  | 指定是否混洗数据集 |
|     | 参数9 | -    | num_shards | 指定分布式训练时将数据集进行划分的分片数 |
|     | 参数10 | -    | shard_id | 指定分布式训练时使用的分片ID号 |
|     | 参数11 | -    | cache | 指定单节点数据缓存服务 |

## 代码示例

```python
# PyTorch
import torchtext.datasets as datasets

root = "/path/to/dataset_root/"
train_iter, valid_iter, test_iter = datasets.IWSLT2016(root, split=('train', 'valid', 'test'))
data = next(iter(train_iter))

# MindSpore
import mindspore.dataset as ds

# Download IWSLT2016 dataset files, unzip into the following structure
# .
# └── /path/to/dataset_directory/
#      ├── subeval_files
#                └── texts
#                    ├── ar
#                    │    └── en
#                    │        └── ar-en
#                    ├── cs
#                    │    └── en
#                    │        └── cs-en
#                    ├── de
#                    │    └── en
#                    │        └── de-en
#                    │            ├── IWSLT16.TED.dev2010.de-en.de.xml
#                    │            ├── train.tags.de-en.de
#                    │            ├── ...
#                    ├── en
#                    │    ├── ar
#                    │    │   └── en-ar
#                    │    ├── cs
#                    │    │   └── en-cs
#                    │    ├── de
#                    │    │   └── en-de
#                    │    └── fr
#                    │        └── en-fr
#                    └── fr
#                        └── en
#                            └── fr-en2
root = "/path/to/dataset_directory/"
dataset = ds.IWSLT2016Dataset(root, usage='all')
data = next(iter(dataset))
```
