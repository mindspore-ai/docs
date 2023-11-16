# 加载文本数据集

`Linux` `Ascend` `GPU` `CPU` `数据准备` `初级` `中级` `高级`

[![查看源文件](../_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/tutorials/training/source_zh_cn/use/load_dataset_text.md)
&nbsp;&nbsp;
[![查看notebook](../_static/logo_notebook.png)](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/r1.1/mindspore_loading_text_dataset.ipynb)
&nbsp;&nbsp;
[![在线运行](../_static/logo_modelarts.png)](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/notebook/loading?share-url-b64=aHR0cHM6Ly9vYnMuZHVhbHN0YWNrLmNuLW5vcnRoLTQubXlodWF3ZWljbG91ZC5jb20vbWluZHNwb3JlLXdlYnNpdGUvbm90ZWJvb2svbW9kZWxhcnRzL21pbmRzcG9yZV9sb2FkaW5nX3RleHRfZGF0YXNldC5pcHluYg==&image_id=65f636a0-56cf-49df-b941-7d2a07ba8c8c)

## 概述

MindSpore提供的`mindspore.dataset`模块可以帮助用户构建数据集对象，分批次地读取文本数据。同时，在各个数据集类中还内置了数据处理和数据分词算子，使得数据在训练过程中能够像经过pipeline管道的水一样源源不断地流向训练系统，提升数据训练效果。

此外，MindSpore还支持分布式场景数据加载，用户可以在加载数据集时指定分片数目，具体用法参见[数据并行模式加载数据集](https://www.mindspore.cn/tutorial/training/zh-CN/r1.1/advanced_use/distributed_training_ascend.html#id6)。

下面，本教程将简要演示如何使用MindSpore加载和处理文本数据。

## 准备

1. 准备文本数据如下。

    ```text
    Welcome to Beijing!
    北京欢迎您！
    我喜欢English!
    ```

2. 创建`tokenizer.txt`文件并复制文本数据到该文件中，将该文件存放在`./test`路径中，目录结构如下。

    ```text
    └─test
        └─tokenizer.txt
    ```

3. 导入`mindspore.dataset`和`mindspore.dataset.text`模块。

    ```python
    import mindspore.dataset as ds
    import mindspore.dataset.text as text
    ```

## 加载数据集

MindSpore目前支持加载文本领域常用的经典数据集和多种数据存储格式下的数据集，用户也可以通过构建自定义数据集类实现自定义方式的数据加载。各种数据集的详细加载方法，可参考编程指南中[数据集加载](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/dataset_loading.html)章节。

下面演示使用`mindspore.dataset`中的`TextFileDataset`类加载数据集。

1. 配置数据集目录，创建数据集对象。

    ```python
    DATA_FILE = "./test/tokenizer.txt"
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    ```

2. 创建迭代器，通过迭代器获取数据。

    ```python
    for data in dataset.create_dict_iterator(output_numpy=True):
        print(text.to_str(data['text']))
    ```

    获取到分词前的数据：

    ```text
    Welcome to Beijing!
    北京欢迎您！
    我喜欢English!
    ```

## 数据处理

MindSpore目前支持的数据处理算子及其详细使用方法，可参考编程指南中[数据处理](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/pipeline.html)章节。

下面演示构建pipeline，对文本数据集进行混洗和文本替换操作。

1. 对数据集进行混洗。

    ```python
    ds.config.set_seed(58)
    dataset = dataset.shuffle(buffer_size=3)

    for data in dataset.create_dict_iterator(output_numpy=True):
        print(text.to_str(data['text']))
    ```

    输出结果如下：

    ```text
    我喜欢English!
    Welcome to Beijing!
    北京欢迎您！
    ```

2. 对数据集进行文本替换。

    ```python
    replace_op1 = text.RegexReplace("Beijing", "Shanghai")
    replace_op2 = text.RegexReplace("北京", "上海")
    dataset = dataset.map(operations=[replace_op1, replace_op2])

    for data in dataset.create_dict_iterator(output_numpy=True):
        print(text.to_str(data['text']))
    ```

    输出结果如下：

    ```text
    我喜欢English!
    Welcome to Shanghai!
    上海欢迎您！
    ```

## 数据分词

MindSpore目前支持的数据分词算子及其详细使用方法，可参考编程指南中[分词器](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.1/tokenizer.html)章节。

下面演示使用`WhitespaceTokenizer`分词器来分词，该分词是按照空格来进行分词。

1. 创建`tokenizer`。

    ```python
    tokenizer = text.WhitespaceTokenizer()
    ```

2. 执行操作`tokenizer`。

    ```python
    dataset = dataset.map(operations=tokenizer)
    ```

3. 创建迭代器，通过迭代器获取数据。

    ```python
    for data in dataset.create_dict_iterator(output_numpy=True):
        print(text.to_str(data['text']).tolist())
    ```

    获取到分词后的数据：

    ```text
    ['我喜欢English!']
    ['Welcome', 'to', 'Shanghai!']
    ['上海欢迎您！']
    ```
