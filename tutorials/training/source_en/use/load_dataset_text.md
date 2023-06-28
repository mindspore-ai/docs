# Loading Text Dataset

`Linux` `Ascend` `GPU` `CPU` `Data Preparation` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/training/source_en/use/load_dataset_text.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

The `mindspore.dataset` module provided by MindSpore enables users to customize their data fetching strategy from disk. At the same time, data processing and tokenization operators are applied to the data. Pipelined data processing produces a continuous flow of data to the training network, improving overall performance.

In addition, MindSpore supports data loading in distributed scenarios. Users can define the number of shards while loading. For more details, see [Loading the Dataset in Data Parallel Mode](https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/distributed_training_ascend.html#loading-the-dataset-in-data-parallel-mode).

This tutorial briefly demonstrates how to load and process text data using MindSpore.

## Preparations

1. Prepare the following text data.

    ```
    Welcome to Beijing!
    北京欢迎您！
    我喜欢English!
    ```

2. Create the `tokenizer.txt` file, copy the text data to the file, and save the file under `./test` directory. The directory structure is as follow.

    ```
    └─test
        └─tokenizer.txt
    ```

3. Import the `mindspore.dataset` and `mindspore.dataset.text` modules.

    ```python
    import mindspore.dataset as ds
    import mindspore.dataset.text as text
    ```

## Loading Dataset

MindSpore supports loading common datasets in the field of text processing that come in a variety of on-disk formats. Users can also implement custom dataset class to load customized data.

The following tutorial demonstrates loading datasets using the `TextFileDataset` in the `mindspore.dataset` module.

1. Configure the dataset directory as follows and create a dataset object.

    ```python
    DATA_FILE = "./test/tokenizer.txt"
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    ```

2. Create an iterator then obtain data through the iterator.

    ```python
    for data in dataset.create_dict_iterator(output_numpy=True):
        print(text.to_str(data['text']))
    ```

    The output without tokenization:

    ```
    Welcome to Beijing!
    北京欢迎您！
    我喜欢English!
    ```

## Processing Data

The following tutorial demonstrates how to perform data processing such as `SlidingWindow` and `shuffle` after a `dataset` is created.

- **SlidingWindow**

    The following tutorial demonstrates how to use the `SlidingWindow` to slice text data.

    1. Load the text dataset.

        ```python
        inputs = [["大", "家", "早", "上", "好"]]
        dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=False)
        ```

    2. Print the results without any data processing.

        ```python
        for data in dataset.create_dict_iterator(output_numpy=True):
                print(text.to_str(data['text']).tolist())
        ```

        The output is as follows:

        ```
        ['大', '家', '早', '上', '好']
        ```

    3. Perform the data processing operation.

        ```python
        dataset = dataset.map(operations=text.SlidingWindow(2, 0), input_columns=["text"])
        ```

    4. Print the results after data processing.

        ```python
        for data in dataset.create_dict_iterator(output_numpy=True):
                print(text.to_str(data['text']).tolist())
        ```

        The output is as follows:

        ```
        [['大', '家'],
         ['家', '早'],
         ['早', '上'],
         ['上', '好']]
        ```

- **shuffle**

    The following tutorial demonstrates how to shuffle text data while loading a dataset.

    1. Load and shuffle the text dataset.

        ```python
        inputs = ["a", "b", "c", "d"]
        dataset = ds.NumpySlicesDataset(inputs, column_names=["text"], shuffle=True)
        ```

    2. Print the results after performing `shuffle`.

        ```python
        for data in dataset.create_dict_iterator(output_numpy=True):
                print(text.to_str(data['text']).tolist())
        ```

        The output is as follows:

        ```
        c
        a
        d
        b
        ```

## Tokenization

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
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            token = text.to_str(i['text']).tolist()
            print(token)
    ```

    The output after tokenization is as follows:

    ```
    ['Welcome', 'to', 'Beijing!']
    ['北京欢迎您！']
    ['我喜欢English!']
    ```
