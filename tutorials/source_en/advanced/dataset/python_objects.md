<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_en/advanced/dataset/python_objects.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

# Supporting Python Objects in Dataset Pipeline

Dataset pipeline accepts Python dictionaries as a distinct input type for some operations. The main difference compared to other types is that Python dictionaries will not be converted to a C++ equivalent, and instead a reference will be maintained in the pipeline.

Note that while currently only `dict` objects are supported and can be passed to the pipeline, you may wrap other Python types inside a dictionary and send them to Dataset pipeline to achieve the same behavior.

## Handling Python `dict` Objects

### Sending Python `dict` to Dataset Pipeline

Sending Python `dict` objects to the Dataset pipeline is possible through different methods:

1. using a `GeneratorDataset`, the user can customize it to return a `dict` object, or
2. within a pyfunc used in a `map` operation, the user can customize it to return a `dict` object, or
3. similarly, customize the `per_batch_map` function of a `batch` operation to return a `dict` object.

### Consuming Python `dict` from Dataset Pipeline

When sending data to the model (or using a standalone iterator), Dataset pipeline attempts to convert all _values_ inside `dict` objects to NumPy arrays (and later to `Tensor` if `output_numpy` is set to `False`). This means that all numerical and string values will be converted to arrays.

Note that this step will be applied recursively to all _values_ inside nested dictionaries as well as all _elements_ inside lists and tuples.

For other types that cannot be converted to NumPy arrays (such as class objects), they are passed directly.

> This feature is currently only supported when `dataset_sink_mode` is set to `False`.

## GeneratorDataset

Each dictionary is considered as one column, whereas there is no limit on the internal items it stores (number and type of keys/values).
Here is an example of sending `dict` objects to the Dataset pipeline using `GeneratorDataset`:

```python
import mindspore.dataset as ds

def my_generator():
    for i in range(5):
        col1 = {"number": i, "square": i ** 2}
        col2 = i
        yield (col1, col2)

data = ds.GeneratorDataset(source=my_generator, column_names=["col1", "col2"])

for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(d)
```

Output:

```python
{'col1': {'number': array(0), 'square': array(0)}, 'col2': array(0, dtype=int64)}
{'col1': {'number': array(1), 'square': array(1)}, 'col2': array(1, dtype=int64)}
{'col1': {'number': array(2), 'square': array(4)}, 'col2': array(2, dtype=int64)}
{'col1': {'number': array(3), 'square': array(9)}, 'col2': array(3, dtype=int64)}
{'col1': {'number': array(4), 'square': array(16)}, 'col2': array(4, dtype=int64)}
```

## Map Operation

Similar to `GeneratorDataset`, each `dict` object is treated as one column and there is no limitation on its content.

> Except for user-defined functions, none of the existing Dataset pipeline transforms (`mindspore.dataset.transforms`) support inputs of type `dict`.

Here is an example of adding `dict` objects to the Dataset pipeline using `map` operation and a pyfunc:

```python
import mindspore.dataset as ds

def my_generator():
    for i in range(5):
        yield i

def my_pyfunc(col1):
    new_col1 = {"original_col1": col1, "square": col1 ** 2}
    return new_col1

data = ds.GeneratorDataset(source=my_generator, column_names=["col1"])
data = data.map(operations=my_pyfunc, input_columns=["col1"])

for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(d)
```

Output:

```python
{'col1': {'original_col1': array(0), 'square': array(0)}}
{'col1': {'original_col1': array(1), 'square': array(1)}}
{'col1': {'original_col1': array(2), 'square': array(4)}}
{'col1': {'original_col1': array(3), 'square': array(9)}}
{'col1': {'original_col1': array(4), 'square': array(16)}}
```

## Batch Operation

When Batch operation is invoked on a dataset with a column containing dictionary objects, Dataset pipeline attempts to group corresponding values inside the dictionaries together. Thus, it is necessary for all the dictionaries to have identical keys.

If the user has provided a `per_batch_map` function, corresponding items inside the dictionaries (with respect to each key) will be grouped into Python lists.

If the user has not provided a `per_batch_map` function, corresponding items inside the dictionaries will be grouped into a NumPy array and the result of the Batch operation (for that column) will be one dictionary where all values are NumPy arrays. If such conversion results in an array of type `np.object_`, due to limitations on the model training side, an error message will be shown to the user and the Dataset pipeline terminates.

Also, note that if the `per_batch_map` returns a single `dict` for a column, it must return dictionaries for other columns as well.

Here is an example of adding `dict` objects to the Dataset pipeline using `batch` operation with a `per_batch_map` function:

```python
import numpy as np
import mindspore.dataset as ds

def my_generator():
    for i in range(15):
        yield i

def my_per_batch_map(col1, batch_info):
    new_col1 = {"original_col1": col1, "index": np.arange(5)}
    return (new_col1,)

data = ds.GeneratorDataset(source=my_generator, column_names=["col1"])
data = data.batch(batch_size=5, per_batch_map=my_per_batch_map)

for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(d)
```

Output:

```python
{'col1': {'original_col1': [array(0), array(1), array(2), array(3), array(4)], 'index': array([0, 1, 2, 3, 4])}}
{'col1': {'original_col1': [array(5), array(6), array(7), array(8), array(9)], 'index': array([0, 1, 2, 3, 4])}}
{'col1': {'original_col1': [array(10), array(11), array(12), array(13), array(14)], 'index': array([0, 1, 2, 3, 4])}}
```

The following is a complex example demonstrating how to use a Python dictionary object with `GeneratorDataset`, `map` operation with a pyfunc and `batch` operation with no `per_batch_map` function. Note that since there is no `per_batch_map` in this code, a single NumPy array is returned for "powers" key.

```python
import numpy as np
import mindspore.dataset as ds

def my_generator():
    for i in range(5):
        col1 = {"nested_dict": {"powers": np.power(i, [1, 2, 3])}}
        col2 = i
        yield (col1, col2)

def my_pyfunc(col1):
    assert isinstance(col1, dict)
    new_col1 = col1["nested_dict"]
    return new_col1

data = ds.GeneratorDataset(source=my_generator, column_names=["col1", "col2"])
data = data.map(operations=my_pyfunc, input_columns=["col1"])
data = data.batch(batch_size=5)

for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(d)
```

Output:

```python
{'col1': {'powers': array([[ 0,  0,  0],
       [ 1,  1,  1],
       [ 2,  4,  8],
       [ 3,  9, 27],
       [ 4, 16, 64]])}, 'col2': array([0, 1, 2, 3, 4], dtype=int64)}
```
