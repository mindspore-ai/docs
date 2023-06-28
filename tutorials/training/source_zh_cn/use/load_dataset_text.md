# 加载文本数据集

`Linux` `Ascend` `GPU` `CPU` `数据准备` `初级` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_zh_cn/use/load_dataset_text.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

MindSpore提供的`mindspore.dataset`模块可以帮助用户构建数据集对象，分批次地读取文本数据。同时，在各个数据集类中还内置了数据处理和数据分词算子，使得数据在训练过程中能够像经过pipeline管道的水一样源源不断地流向训练系统，提升数据训练效果。

此外，MindSpore还支持分布式场景数据加载，用户可以在加载数据集时指定分片数目，具体用法参见[数据并行模式加载数据集](https://www.mindspore.cn/tutorial/training/zh-CN/r1.0/advanced_use/distributed_training_ascend.html#id6)。

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

MindSpore目前支持加载文本领域常用的经典数据集和多种数据存储格式下的数据集，用户也可以通过构建自定义数据集类实现自定义方式的数据加载。各种数据集的详细加载方法，可参考编程指南中[数据集加载](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.0/dataset_loading.html)章节。

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

MindSpore目前支持的数据处理算子及其详细使用方法，可参考编程指南中[数据处理](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.0/pipeline.html)章节。

在生成`dataset`对象后可对其进行数据处理操作，比如`SlidingWindow`、`shuffle`等。

- **SlidingWindow**

    下面演示使用`SlidingWindow`对文本数据进行切片操作。

    1. 加载数据集。

        ```python
        inputs = [["大", "家", "早", "上", "好"]]
        dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=False)
        ```

    2. 原始数据输出效果。

        ```python
        for data in dataset.create_dict_iterator(output_numpy=True):
                print(text.to_str(data['text']).tolist())
        ```

        输出结果如下：

        ```text
        ['大', '家', '早', '上', '好']
        ```

    3. 执行操作。

        ```python
        dataset = dataset.map(operations=text.SlidingWindow(2, 0), input_columns=["text"])
        ```

    4. 执行之后输出效果。

        ```python
        for data in dataset.create_dict_iterator(output_numpy=True):
                print(text.to_str(data['text']).tolist())
        ```

        输出结果如下：

        ```text
        [['大', '家'],
         ['家', '早'],
         ['早', '上'],
         ['上', '好']]
        ```

- **shuffle**

    下面演示在加载数据集时使用`shuffle`对文本数据进行混洗操作。

    1. 加载数据集。

        ```python
        inputs = ["a", "b", "c", "d"]
        dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=True)
        ```

    2. 数据输出效果。

        ```python
        for data in dataset.create_dict_iterator(output_numpy=True):
                print(text.to_str(data['text']).tolist())
        ```

        输出结果如下：

        ```text
        c
        a
        d
        b
        ```

## 数据分词

MindSpore目前支持的数据分词算子及其详细使用方法，可参考编程指南中[分词器](https://www.mindspore.cn/doc/programming_guide/zh-CN/r1.0/tokenizer.html)章节。

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
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            token = text.to_str(i['text']).tolist()
            print(token)
    ```

    获取到分词后的数据：

    ```text
    ['Welcome', 'to', 'Beijing!']
    ['北京欢迎您！']
    ['我喜欢English!']
    ```
