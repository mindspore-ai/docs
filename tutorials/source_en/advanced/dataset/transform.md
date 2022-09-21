# Data Processing

<a href="https://gitee.com/mindspore/docs/blob/r1.9/tutorials/source_en/advanced/dataset/transform.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png"></a>

Data is the basis of deep learning. Good data input plays an important role in the deep neural network training. Data processing is performed on a loaded dataset before training, so that problems such as an excessively large data volume and uneven sample distribution can be resolved, thereby obtaining data input that is more favorable to a training result.

Each dataset class of MindSpore provides multiple data processing operations for users. Users can build a data processing pipeline to define the data processing operations to be used. During training, data can continuously flow to the training system through the data processing pipeline.

MindSpore supports common data processing operations, such as data `shuffle`, `batch`, `repeat`, and `concat`.

> For more data processing operations, see [API](https://www.mindspore.cn/docs/en/r1.9/api_python/mindspore.dataset.html).

## Data Processing Operations

### shuffle

The shuffle operation randomly disrupts the data sequence and shuffles datasets.

A larger `buffer_size` value indicates a higher degree of data shuffling and consumes more time and computing resources.

![shuffle](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/source_zh_cn/advanced/dataset/images/op_shuffle.png)

The following example first builds a random dataset, then shuffles it, and finally shows the results before and after data shuffling.

```python
import numpy as np
import mindspore.dataset as ds

ds.config.set_seed(0)

def generator_func():
    """Define a function for generating a dataset."""
    for i in range(5):
        yield (np.array([i, i+1, i+2]),)

# Generate a dataset.
dataset = ds.GeneratorDataset(generator_func, ["data"])
for data in dataset.create_dict_iterator():
    print(data)

print("------ after processing ------")

# Shuffle data.
dataset = dataset.shuffle(buffer_size=2)
for data in dataset.create_dict_iterator():
    print(data)
```

```text
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])}
    ------ after processing ------
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])}
```

According to the preceding result, the data sequence is disrupted after the `shuffle` operation.

### batch

The batch operation groups datasets into batches and inputs them to the training system. This reduces the number of training epochs and accelerates the training process.

![batch](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/source_zh_cn/advanced/dataset/images/op_batch.png)

The following example builds a dataset, and then displays the dataset batching results with or without dropping redundant data. The batch size is 2.

```python
import numpy as np
import mindspore.dataset as ds

def generator_func():
    """Define a function for generating a dataset."""
    for i in range(5):
        yield (np.array([i, i+1, i+2]),)

dataset = ds.GeneratorDataset(generator_func, ["data"])
for data in dataset.create_dict_iterator():
    print(data)

# Batch datasets without dropping redundant data.
dataset = ds.GeneratorDataset(generator_func, ["data"])
dataset = dataset.batch(batch_size=2, drop_remainder=False)
print("------not drop remainder ------")
for data in dataset.create_dict_iterator():
    print(data)

# Batch datasets by dropping redundant data.
dataset = ds.GeneratorDataset(generator_func, ["data"])
dataset = dataset.batch(batch_size=2, drop_remainder=True)
print("------ drop remainder ------")
for data in dataset.create_dict_iterator():
    print(data)
```

```text
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])}
    ------not drop remainder ------
    {'data': Tensor(shape=[2, 3], dtype=Int64, value=
    [[0, 1, 2],
     [1, 2, 3]])}
    {'data': Tensor(shape=[2, 3], dtype=Int64, value=
    [[2, 3, 4],
     [3, 4, 5]])}
    {'data': Tensor(shape=[1, 3], dtype=Int64, value=
    [[4, 5, 6]])}
    ------ drop remainder ------
    {'data': Tensor(shape=[2, 3], dtype=Int64, value=
    [[0, 1, 2],
     [1, 2, 3]])}
    {'data': Tensor(shape=[2, 3], dtype=Int64, value=
    [[2, 3, 4],
     [3, 4, 5]])}
```

According to the preceding result, there are five pieces of data. Every two of them are grouped into one batch. If redundant data is not dropped, the datasets are grouped into three batches. If redundant data is dropped, the datasets are grouped into two batches. The last data is dropped.

### repeat

The repeat operation repeats the dataset to expand the data volume.The sequence of `repeat` and `batch` affect the number of training batches. You are advised to place `repeat` after `batch`.

![repeat](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/source_zh_cn/advanced/dataset/images/op_repeat.png)

The following example builds a random dataset, repeats it twice, and finally shows the repeated data result.

```python
import numpy as np
import mindspore.dataset as ds

def generator_func():
    """Define a function for generating a dataset."""
    for i in range(5):
        yield (np.array([i, i+1, i+2]),)

# Generate a dataset.
dataset = ds.GeneratorDataset(generator_func, ["data"])
for data in dataset.create_dict_iterator():
    print(data)

print("------ after processing ------")

# Repeat data.
dataset = dataset.repeat(count=2)
for data in dataset.create_dict_iterator():
    print(data)
```

```text
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])}
    ------ after processing ------
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])}
```

According to the preceding result, the dataset copy is placed at the end of the original dataset.

### zip

The zip operation combines columns of two datasets to obtain a new dataset. Note that:

1. If the column names in the two datasets are the same, the two datasets are not combined. Therefore, pay attention to column names.
2. If the number of rows in the two datasets is different, the number of rows after combination is the same as the smaller number of rows.

![zip](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/source_zh_cn/advanced/dataset/images/op_zip.png)

The following example builds two random datasets with different numbers of samples, combines columns, and displays the combined data result.

```python
import numpy as np
import mindspore.dataset as ds

def generator_func():
    """Define a function for generating a dataset.1"""
    for i in range(7):
        yield (np.array([i, i+1, i+2]),)

def generator_func2():
    """Define a function for generating a dataset.2"""
    for _ in range(4):
        yield (np.array([1, 2]),)

print("------ data1 ------")
dataset1 = ds.GeneratorDataset(generator_func, ["data1"])
for data in dataset1.create_dict_iterator():
    print(data)

print("------ data2 ------")
dataset2 = ds.GeneratorDataset(generator_func2, ["data2"])
for data in dataset2.create_dict_iterator():
    print(data)

print("------ data3 ------")

# Perform the zip operation on dataset 1 and dataset 2 to generate dataset 3.
dataset3 = ds.zip((dataset1, dataset2))
for data in dataset3.create_dict_iterator():
    print(data)
```

```text
    ------ data1 ------
    {'data1': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])}
    {'data1': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
    {'data1': Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])}
    {'data1': Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])}
    {'data1': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])}
    {'data1': Tensor(shape=[3], dtype=Int64, value= [5, 6, 7])}
    {'data1': Tensor(shape=[3], dtype=Int64, value= [6, 7, 8])}
    ------ data2 ------
    {'data2': Tensor(shape=[2], dtype=Int64, value= [1, 2])}
    {'data2': Tensor(shape=[2], dtype=Int64, value= [1, 2])}
    {'data2': Tensor(shape=[2], dtype=Int64, value= [1, 2])}
    {'data2': Tensor(shape=[2], dtype=Int64, value= [1, 2])}
    ------ data3 ------
    {'data1': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2]), 'data2': Tensor(shape=[2], dtype=Int64, value= [1, 2])}
    {'data1': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3]), 'data2': Tensor(shape=[2], dtype=Int64, value= [1, 2])}
    {'data1': Tensor(shape=[3], dtype=Int64, value= [2, 3, 4]), 'data2': Tensor(shape=[2], dtype=Int64, value= [1, 2])}
    {'data1': Tensor(shape=[3], dtype=Int64, value= [3, 4, 5]), 'data2': Tensor(shape=[2], dtype=Int64, value= [1, 2])}
```

According to the preceding result, columns of dataset 1 and dataset 2 are combined to obtain dataset 3. The number of columns of dataset 3 is the sum of columns of dataset 1 and dataset 2. The number of rows of dataset 3 is the same as that of dataset 2 which is the smaller one. The extra rows in dataset 1 are dropped.

### concat

The concat operation concatenates rows of two datasets to obtain a new dataset. Note that the column names, column data types, and column data arrangement in the input datasets must be the same.

![concat](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/source_zh_cn/advanced/dataset/images/op_concat.png)

The following example builds two random datasets, concatenates rows, and displays the concatenated data result. Note that using the `+` operator can achieve the same effect.

```python
import numpy as np
import mindspore.dataset as ds

def generator_func():
    """Define a function for generating a dataset.1"""
    for _ in range(2):
        yield (np.array([0, 0, 0]),)

def generator_func2():
    """Define a function for generating a dataset.2"""
    for _ in range(2):
        yield (np.array([1, 2, 3]),)

# Generate a dataset.1
dataset1 = ds.GeneratorDataset(generator_func, ["data"])
print("data1:")
for data in dataset1.create_dict_iterator():
    print(data)

# Generate a dataset.2
dataset2 = ds.GeneratorDataset(generator_func2, ["data"])
print("data2:")
for data in dataset2.create_dict_iterator():
    print(data)

# Perform the concat operation on dataset 1 and dataset 2 to generate dataset 3.
dataset3 = dataset1.concat(dataset2)
print("data3:")
for data in dataset3.create_dict_iterator():
    print(data)
```

```text
    data1:
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 0, 0])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 0, 0])}
    data2:
    {'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
    data3:
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 0, 0])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 0, 0])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
```

According to the preceding result, rows of dataset 1 and dataset 2 are concatenated to obtain dataset 3. The number of columns of dataset 3 is the same as that of dataset 1 and dataset 2, and the number of rows of dataset 3 is the sum of that of dataset 1 and dataset 2.

### map

The map operation applies a specified function to data in a specified column of a dataset to implement data mapping.

You can customize mapping functions or directly use the functions in `c_transforms` or `py_transforms` to augment image and text data.

![map](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/tutorials/source_en/advanced/dataset/images/op_map.png)

The following example builds a random dataset, defines the mapping function for doubling data, and applies it to the dataset to compare the data results before and after the mapping.

```python
import numpy as np
import mindspore.dataset as ds

def generator_func():
    """Define a function for generating a dataset."""
    for i in range(5):
        yield (np.array([i, i+1, i+2]),)

def pyfunc(x):
    """Define the operation on data."""
    return x*2

# Generate a dataset.
dataset = ds.GeneratorDataset(generator_func, ["data"])

# Display the dataset generated above.
for data in dataset.create_dict_iterator():
    print(data)

print("------ after processing ------")

# Perform the map operation on the dataset. The operation function is pyfunc.
dataset = dataset.map(operations=pyfunc, input_columns=["data"])

# Display the data set after the map operation.
for data in dataset.create_dict_iterator():
    print(data)
```

```text
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 1, 2])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [1, 2, 3])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [2, 3, 4])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [3, 4, 5])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [4, 5, 6])}
    ------ after processing ------
    {'data': Tensor(shape=[3], dtype=Int64, value= [0, 2, 4])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [2, 4, 6])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [4, 6, 8])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [ 6,  8, 10])}
    {'data': Tensor(shape=[3], dtype=Int64, value= [ 8, 10, 12])}
```

According to the preceding result, after the map operation is performed and the `pyfunc` function is applied to the dataset, each data in the dataset is multiplied by 2.
