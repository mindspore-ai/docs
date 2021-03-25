# Loading Text Dataset

`Linux` `Ascend` `GPU` `CPU` `Data Preparation` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Loading Text Dataset](#loading-text-dataset)
    - [Overview](#overview)
    - [Preparations](#preparations)
    - [Loading Dataset](#loading-dataset)
    - [Processing Data](#processing-data)
    - [Tokenization](#tokenization)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.2/tutorials/training/source_en/use/load_dataset_text.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

The `mindspore.dataset` module provided by MindSpore enables users to customize their data fetching strategy from disk. At the same time, data processing and tokenization operators are applied to the data. Pipelined data processing produces a continuous flow of data to the training network, improving overall performance.

In addition, MindSpore supports data loading in distributed scenarios. Users can define the number of shards while loading. For more details, see [Loading the Dataset in Data Parallel Mode](https://www.mindspore.cn/tutorial/training/en/r1.2/advanced_use/distributed_training_ascend.html#loading-the-dataset-in-data-parallel-mode).

This tutorial briefly demonstrates how to load and process text data using MindSpore.

## Preparations

1. Prepare the following text data.

    ```text
    Welcome to Beijing!
    北京欢迎您！
    我喜欢English!
    ```

2. Create the `tokenizer.txt` file, copy the text data to the file, and save the file under `./datasets` directory. The directory structure is as follow.

    ```python
    import os

    if not os.path.exists('./datasets'):
        os.mkdir('./datasets')
    file_handle=open('./datasets/tokenizer.txt',mode='w')
    file_handle.write('Welcome to Beijing \n北京欢迎您！ \n我喜欢English! \n')
    file_handle.close()
    ```

    ```bash
    tree ./datasets
    ```

    ```text
    ./datasets
    └── tokenizer.txt

    0 directories, 1 file
    ```

3. Import the `mindspore.dataset` and `mindspore.dataset.text` modules.

    ```python
    import mindspore.dataset as ds
    import mindspore.dataset.text as text
    ```

## Loading Dataset

MindSpore supports loading common datasets in the field of text processing that come in a variety of on-disk formats. Users can also implement custom dataset class to load customized data. For detailed loading methods of various datasets, please refer to the [Loading Dataset](https://www.mindspore.cn/doc/programming_guide/en/r1.2/dataset_loading.html) chapter in the Programming Guide.

The following tutorial demonstrates loading datasets using the `TextFileDataset` in the `mindspore.dataset` module.

1. Configure the dataset directory as follows and create a dataset object.

    ```python
    DATA_FILE = "./datasets/tokenizer.txt"
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    ```

2. Create an iterator then obtain data through the iterator.

    ```python
    for data in dataset.create_dict_iterator(output_numpy=True):
        print(text.to_str(data['text']))
    ```

    The output without tokenization:

    ```text
    Welcome to Beijing!
    北京欢迎您！
    我喜欢English!
    ```

## Processing Data

For the data processing operators currently supported by MindSpore and their detailed usage methods, please refer to the [Processing Data](https://www.mindspore.cn/doc/programming_guide/en/r1.2/pipeline.html) chapter in the Programming Guide

The following tutorial demonstrates how to construct a pipeline and perform operations such as `shuffle` and `RegexReplace` on the text dataset.

1. Shuffle the dataset.

    ```python
    ds.config.set_seed(58)
    dataset = dataset.shuffle(buffer_size=3)

    for data in dataset.create_dict_iterator(output_numpy=True):
        print(text.to_str(data['text']))
    ```

    The output is as follows:

    ```text
    我喜欢English!
    Welcome to Beijing!
    北京欢迎您！
    ```

2. Perform `RegexReplace` on the dataset.

    ```python
    replace_op1 = text.RegexReplace("Beijing", "Shanghai")
    replace_op2 = text.RegexReplace("北京", "上海")
    dataset = dataset.map(operations=replace_op1)
    dataset = dataset.map(operations=replace_op2)

    for data in dataset.create_dict_iterator(output_numpy=True):
        print(text.to_str(data['text']))
    ```

    The output is as follows:

    ```text
    我喜欢English!
    Welcome to Shanghai!
    上海欢迎您！
    ```

## Tokenization

For the data tokenization operators currently supported by MindSpore and their detailed usage methods, please refer to the [Tokenizer](https://www.mindspore.cn/doc/programming_guide/en/r1.2/tokenizer.html) chapter in the Programming Guide.

The following tutorial demonstrates how to use the `WhitespaceTokenizer` to tokenize words with space.

1. Create a `tokenizer`.

    ```python
    tokenizer = text.WhitespaceTokenizer()
    ```

2. Apply the `tokenizer`.

    ```python
    dataset = dataset.map(operations=tokenizer)
    ```

3. Create an iterator and obtain data through the iterator.

    ```python
    for data in dataset.create_dict_iterator(output_numpy=True):
        print(text.to_str(data['text']).tolist())
    ```

    The output after tokenization is as follows:

    ```text
    ['我喜欢English!']
    ['Welcome', 'to', 'Shanghai!']
    ['上海欢迎您！']
    ```
