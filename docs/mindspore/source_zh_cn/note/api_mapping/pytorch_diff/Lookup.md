# 比较与torchtext.data.functional.numericalize_tokens_from_iterator的差异

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/Lookup.md)

## torchtext.data.functional.numericalize_tokens_from_iterator

```python
torchtext.data.functional.numericalize_tokens_from_iterator(
    vocab,
    iterator,
    removed_tokens=None
)
```

更多内容详见[torchtext.data.functional.numericalize_tokens_from_iterator](https://pytorch.org/text/0.9.0/data_functional.html#numericalize-tokens-from-iterator)。

## mindspore.dataset.text.Lookup

```python
class mindspore.dataset.text.Lookup(
    vocab,
    unknown_token=None,
    data_type=mstype.int32
)
```

更多内容详见[mindspore.dataset.text.Lookup](https://mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/dataset_text/mindspore.dataset.text.Lookup.html#mindspore.dataset.text.Lookup)。

## 使用方式

PyTorch：从分词迭代器中生成词汇表对应的id列表，输入为词汇与id对应的映射表、词汇迭代器，返回创建好的迭代器对象，可从中获取对应词汇的id。

MindSpore：依据词汇与id的映射表，查找词汇对应的id。

| 分类 | 子类 |PyTorch | MindSpore | 差异 |
| --- | ---   | ---   | ---        |---  |
|参数 | 参数1 | vocab     | vocab     | - |
|     | 参数2 | iterator   |-     | 被查表对象，MindSpore通过另一种方式支持，见示例 |
|     | 参数3 | removed_tokens    |-     | 输出结果时需要移除的分词，MindSpore不支持 |
|     | 参数4 | -   |unknown_token    | 备用词汇，用于要查找的单词不在词汇表时进行替换 |
|     | 参数5 | -   |data_type    | Lookup输出的数据类型 |

## 代码示例

```python
# PyTorch
from torchtext.data.functional import numericalize_tokens_from_iterator

def gen():
    yield ["Sentencepiece", "as", "encode"]

vocab = {'Sentencepiece' : 0, 'encode' : 1, 'as' : 2, 'pieces' : 3}
ids_iter = numericalize_tokens_from_iterator(vocab, gen())
for ids in ids_iter:
    print([num for num in ids])
# Out: [0, 2, 1]


# MindSpore
import mindspore.dataset.text as text

vocab = text.Vocab.from_dict({'Sentencepiece' : 0, 'encode' : 1, 'as' : 2, 'pieces' : 3})
result = text.Lookup(vocab)(["Sentencepiece", "as", "encode"])
print(result)
# Out: [0 2 1]
```
