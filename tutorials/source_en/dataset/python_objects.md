# Supporting Python Objects in Dataset Pipeline

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/dataset/python_objects.md)

Dataset pipeline accepts any Python type as input for some operations(such as user-defined dataset `GeneratorDataset`, user-defined `map` augmentation operation, `batch(per_batch_map=...)`).

To achieve this feature, Dataset pipeline uses Python `dict` to manager different types. The main difference compared to other data types is that Python `dict` will not be converted to a C++ type, and instead a reference will be maintained in the pipeline.

Note that while currently Dataset pipeline supports to recognize `dict` objects, you can also wrap other Python types inside a dictionary and send them to Dataset pipeline to achieve the same behavior. This article describes how to construct dictionary type to Dataset pipeline and acquire through iterator.

## Sending Python `dict` to Dataset Pipeline

Sending Python `dict` objects to the Dataset pipeline is possible through different operations:

1. using a `GeneratorDataset`, the user can customize it to return a `dict` object, or
2. within a Python callable object used in a `map` operation, the user can customize it to return a `dict` object, or
3. similarly, customize the `per_batch_map` function of a `batch` operation to return a `dict` object.

### Processing `dict` with GeneratorDataset

Here is an example of sending `dict` objects to the Dataset pipeline using `GeneratorDataset`.

In the example, `my_generator` returns 2 elements, corresponding to 2 data columns, where the dictionary is considered as `col1`, the first column. The rules for data processing pipelines typically check whether the return value can be converted to a NumPy type, but no typecast check if the return value is a dictionary. Furthermore, there is no limit on the internal items it stores (number and type of keys/values).

```python
import mindspore.dataset as ds

def my_generator():
    for i in range(5):
        col1 = {"number": i, "square": i ** 2}
        col2 = i
        yield col1, col2

data = ds.GeneratorDataset(source=my_generator, column_names=["col1", "col2"])

for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(d)
```

Output:

```text
{'col1': {'number': array(0), 'square': array(0)}, 'col2': array(0, dtype=int64)}
{'col1': {'number': array(1), 'square': array(1)}, 'col2': array(1, dtype=int64)}
{'col1': {'number': array(2), 'square': array(4)}, 'col2': array(2, dtype=int64)}
{'col1': {'number': array(3), 'square': array(9)}, 'col2': array(3, dtype=int64)}
{'col1': {'number': array(4), 'square': array(16)}, 'col2': array(4, dtype=int64)}
```

### Processing `dict` with Map Operation

Similar to `GeneratorDataset`, each `dict` object is treated as one data column and there is no limitation on its content.

> Except for user-defined functions, none of the existing Dataset pipeline transforms (`mindspore.dataset.transforms`, `mindspore.dataset.vision`, etc.) support inputs of type `dict`.

Here is an example of adding `dict` objects to the Dataset pipeline using `map` operation and a user-defined function:

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

```text
{'col1': {'original_col1': array(0), 'square': array(0)}}
{'col1': {'original_col1': array(1), 'square': array(1)}}
{'col1': {'original_col1': array(2), 'square': array(4)}}
{'col1': {'original_col1': array(3), 'square': array(9)}}
{'col1': {'original_col1': array(4), 'square': array(16)}}
```

### Processing `dict` with Batch Operation

When `batch` operation is invoked on a dataset with a column containing dictionary objects, Dataset pipeline attempts to group corresponding values with the same keys inside the dictionaries together. Thus, it is necessary for all the dictionaries to have identical keys.

The result of the `batch` operation (for that column) will be one dictionary where all values are NumPy arrays. If such conversion results in an array of type `np.object_`, due to limitations on the model training side, an error message will be shown to the user and the Dataset pipeline terminates.

The following is a example demonstrating when dictionary object exists in dataset pipeline, how it batches the data of "power" key.

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

print(">>> before batch")
for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(d)

data = data.batch(batch_size=5)

print(">>> after batch")
for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(d)
```

Output:

```text
>>> before batch
{'col1': {'powers': array([0, 0, 0])}, 'col2': array(0, dtype=int64)}
{'col1': {'powers': array([1, 1, 1])}, 'col2': array(1, dtype=int64)}
{'col1': {'powers': array([2, 4, 8])}, 'col2': array(2, dtype=int64)}
{'col1': {'powers': array([3, 9, 27])}, 'col2': array(3, dtype=int64)}
{'col1': {'powers': array([4, 16, 64])}, 'col2': array(4, dtype=int64)}
>>> after batch
{'col1': {'powers': array([[0,  0,  0],
                           [1,  1,  1],
                           [2,  4,  8],
                           [3,  9, 27],
                           [4, 16, 64]])},
 'col2': array([0, 1, 2, 3, 4], dtype=int64)}
```

If the user has provided a `per_batch_map` function, corresponding items inside the dictionaries (with respect to each key) will be grouped into Python lists. Here is an example of adding `dict` objects to the Dataset pipeline using `batch` operation with a `per_batch_map` function:

```python
import numpy as np
import mindspore.dataset as ds

def my_generator():
    for i in range(9):
        yield i

def my_per_batch_map(col1, batch_info):
    new_col1 = {"original_col1": col1, "index": np.arange(3)}
    new_col2 = {"copied_col1": col1}
    return new_col1, new_col2

data = ds.GeneratorDataset(source=my_generator, column_names=["col1"])
data = data.batch(batch_size=3, per_batch_map=my_per_batch_map, output_columns=["col1", "col2"])

for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(d)
```

Output:

```text
{'col1': {'original_col1': [array(0), array(1), array(2)], 'index': array([0, 1, 2])}, 'col2': {'copied_col1': [array(0), array(1), array(2)]}}
{'col1': {'original_col1': [array(3), array(4), array(5)], 'index': array([0, 1, 2])}, 'col2': {'copied_col1': [array(3), array(4), array(5)]}}
{'col1': {'original_col1': [array(6), array(7), array(8)], 'index': array([0, 1, 2])}, 'col2': {'copied_col1': [array(6), array(7), array(8)]}}
```

## Getting Python `dict` from Dataset Pipeline

Directly iterating through the dataset object can obtain dictionary type data. When using an iterator to retrieve data, the data processing pipeline will attempt to convert all values inside `dict` objects to Tensor type (if `output_numpy` is set to `True`, it will be converted to NumPy arrays).

Note that this step will be applied recursively to all values inside nested dictionaries as well as all elements inside lists and tuples. For those types that cannot be converted to Tensor/NumPy arrays (such as class objects), they will be passed directly to model. If model can not recognize these types, error will be raised.

Here is an example shows how to acquire `dict` data from pipeline.

```python
import numpy as np
import mindspore.dataset as ds

def my_generator():
    for i in range(5):
        col1 = {"my_data": np.array(i)}
        col2 = i
        yield col1, col2

data = ds.GeneratorDataset(source=my_generator, column_names=["col1", "col2"])

print(">>> Iter dataset with converting all data to Tensor")
for d in data.create_dict_iterator(num_epochs=1):
    print(d)

print(">>> Iter dataset with converting all data to Numpy")
for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
    print(d)
```

Output:

```text
>>> Iter dataset with converting all data to Tensor
{'col1': {'my_data': Tensor(shape=[], dtype=Int64, value= 0)}, 'col2': Tensor(shape=[], dtype=Int64, value= 0)}
{'col1': {'my_data': Tensor(shape=[], dtype=Int64, value= 1)}, 'col2': Tensor(shape=[], dtype=Int64, value= 1)}
{'col1': {'my_data': Tensor(shape=[], dtype=Int64, value= 2)}, 'col2': Tensor(shape=[], dtype=Int64, value= 2)}
{'col1': {'my_data': Tensor(shape=[], dtype=Int64, value= 3)}, 'col2': Tensor(shape=[], dtype=Int64, value= 3)}
{'col1': {'my_data': Tensor(shape=[], dtype=Int64, value= 4)}, 'col2': Tensor(shape=[], dtype=Int64, value= 4)}
>>> Iter dataset with converting all data to Numpy
{'col1': {'my_data': array(0)}, 'col2': array(0, dtype=int64)}
{'col1': {'my_data': array(1)}, 'col2': array(1, dtype=int64)}
{'col1': {'my_data': array(2)}, 'col2': array(2, dtype=int64)}
{'col1': {'my_data': array(3)}, 'col2': array(3, dtype=int64)}
{'col1': {'my_data': array(4)}, 'col2': array(4, dtype=int64)}
```

In the model training/inference scenario, there are the following constraints when obtaining `dict` data from the data pipeline.

- In data sink mode, since the data sink channel currently cannot support dictionary type data, sending dictionary type data to it will cause errors. Therefore, it is suggested to consider turning off the data sink mode (`dataset_sink_mode=False`), or expanding dictionary type data into list or tuple type data at the last data processing node, for example:

  ```python
  import numpy as np
  import mindspore.dataset as ds

  def my_generator():
      for i in range(5):
          col1 = {'my_data': np.array(i), 'my_data2': np.array(i + 1)}
          yield col1

  data = ds.GeneratorDataset(source=my_generator, column_names=['col1'])

  print('>>> get data in dict type')
  for d in data:
      print(d)

  def dict_to_tuple(d):
      return tuple([i for i in d.values()])

  # flatten the dict object bedfore it passed into network
  data = data.map(dict_to_tuple, input_columns=['col1'], output_columns=['my_data', 'my_data2'])

  print('>>> get data in sequence type')
  for d in data:
      print(d)
  ```

  Output:

  ```text
  >>> get data in dict type
  [{'my_data': Tensor(shape=[], dtype=Int64, value= 0), 'my_data2': Tensor(shape=[], dtype=Int64, value= 1)}]
  [{'my_data': Tensor(shape=[], dtype=Int64, value= 1), 'my_data2': Tensor(shape=[], dtype=Int64, value= 2)}]
  [{'my_data': Tensor(shape=[], dtype=Int64, value= 2), 'my_data2': Tensor(shape=[], dtype=Int64, value= 3)}]
  [{'my_data': Tensor(shape=[], dtype=Int64, value= 3), 'my_data2': Tensor(shape=[], dtype=Int64, value= 4)}]
  [{'my_data': Tensor(shape=[], dtype=Int64, value= 4), 'my_data2': Tensor(shape=[], dtype=Int64, value= 5)}]
  >>> get data in sequence type
  [Tensor(shape=[], dtype=Int64, value= 0), Tensor(shape=[], dtype=Int64, value= 1)]
  [Tensor(shape=[], dtype=Int64, value= 1), Tensor(shape=[], dtype=Int64, value= 2)]
  [Tensor(shape=[], dtype=Int64, value= 2), Tensor(shape=[], dtype=Int64, value= 3)]
  [Tensor(shape=[], dtype=Int64, value= 3), Tensor(shape=[], dtype=Int64, value= 4)]
  [Tensor(shape=[], dtype=Int64, value= 4), Tensor(shape=[], dtype=Int64, value= 5)]
  ```

- In non data sink mode, there is no limit. Pay attention to whether the types stored in the dictionary can be recognized and processed by the model.
