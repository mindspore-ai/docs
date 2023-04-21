<a href="https://gitee.com/mindspore/docs/blob/r2.0/tutorials/source_zh_cn/advanced/dataset/python_objects.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source.png"></a>

# 数据处理管道支持Python对象

数据处理管道中的特定操作（如自定义数据集`GeneratorDataset`、自定义`map`增强操作、自定义`batch(per_batch_map=...)`）支持任意Python类型对象作为输入。为了支持此特性，数据管道使用了Python(`dict`)字典去管理不同类型的对象。与其他类型相比，Python字典类型在数据管道中不会被转换成C++中的类型，而是以引用的形式保留在数据管道中。

注意，虽然目前数据管道只新增了识别字典类型的对象，但并不限制字典中的对象内容，因此也可以将其他Python类型封装进字典中并传入到数据处理管道中，以达到支持任意Python对象的目的。因此此教程主要介绍如何构造字典类型的数据输入到数据管道，并在迭代器中取得数据。

## 构造Python字典到数据处理管道

将字典输入到数据处理管道中可在以下几个操作中实现：

1. 自定义数据集`GeneratorDataset`，用户将组织好的字典以返回值的形式输入到数据处理管道中；
2. 自定义`map`增强操作，用户可以定义Python可调用对象，在该对象中返回字典数据；
3. 自定义`batch(per_batch_map=...)`操作，用户在`batch`操作的`per_batch_map`中处理并返回字典数据。

### 自定义数据集`GeneratorDataset`处理字典对象

下面这个例子展示了如何通过`GeneratorDataset`将字典对象传送到数据处理管道。

示例中的`my_generator`返回了2个元素，分别对应2个数据列，其中字典被视为其中一列，即`col1`。特别的，数据处理管道的规则一般会检查返回值是否可以被转换为NumPy类型，但若返回值为字典则会例外，且字典中存储的元素没有限制（包括键/值的数量和的类型）。

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

输出:

```python
{'col1': {'number': array(0), 'square': array(0)}, 'col2': array(0, dtype=int64)}
{'col1': {'number': array(1), 'square': array(1)}, 'col2': array(1, dtype=int64)}
{'col1': {'number': array(2), 'square': array(4)}, 'col2': array(2, dtype=int64)}
{'col1': {'number': array(3), 'square': array(9)}, 'col2': array(3, dtype=int64)}
{'col1': {'number': array(4), 'square': array(16)}, 'col2': array(4, dtype=int64)}
```

### 自定义`map`增强操作处理字典对象

与`GeneratorDataset`相同，每个字典对象被看作一个数据列，且其中的元素没有限制。

> 除了用户自定义函数以外，现有的数据处理管道变换操作(`mindspore.dataset.transforms`、`mindspore.dataset.vision`等)均不支持字典类型的输入。

这个例子说明如何通过`map`操作和自定义Python方法，将字典类型加入到数据处理管道中：

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

输出:

```python
{'col1': {'original_col1': array(0), 'square': array(0)}}
{'col1': {'original_col1': array(1), 'square': array(1)}}
{'col1': {'original_col1': array(2), 'square': array(4)}}
{'col1': {'original_col1': array(3), 'square': array(9)}}
{'col1': {'original_col1': array(4), 'square': array(16)}}
```

### `batch`操作处理字典对象

当对数据集使用`batch`操作时，如果有一个数据列中包含有字典对象，数据处理管道会将多组样本中的字典的相同键组合在一起。因此对数据进行`batch`操作前，确保所有的字典对象都必须具有相同的键。

`batch`操作的结果（对于该列）也将是一个字典，其中所有值都是NumPy数组。如果这种转换产生了`np.object_`类型的数组，由于模型训练侧的限制，将向用户显示一条错误消息并且终止数据处理管道。

下面展示了当数据管道中存在Python字典时，`batch`操作是如何把字典中"power"键的元素组合起来的。

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

输出:

```python
>>> before batch
{'col1': {'powers': array([0, 0, 0])}, 'col2': array(0, dtype=int64)}
{'col1': {'powers': array([1, 1, 1])}, 'col2': array(1, dtype=int64)}
{'col1': {'powers': array([2, 4, 8])}, 'col2': array(2, dtype=int64)}
{'col1': {'powers': array([ 3,  9, 27])}, 'col2': array(3, dtype=int64)}
{'col1': {'powers': array([ 4, 16, 64])}, 'col2': array(4, dtype=int64)}
>>> after batch
{'col1': {'powers': array([[ 0,  0,  0],
                           [ 1,  1,  1],
                           [ 2,  4,  8],
                           [ 3,  9, 27],
                           [ 4, 16, 64]])},
 'col2': array([0, 1, 2, 3, 4], dtype=int64)}
```

如果用户提供了`per_batch_map`函数，字典中的对应元素将根据键分组到Python列表中。这个例子说明如何通过`batch`操作和`per_batch_map`方法，将字典类型加入到数据处理管道中：

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

输出:

```python
{'col1': {'original_col1': [array(0), array(1), array(2)], 'index': array([0, 1, 2])}, 'col2': {'copied_col1': [array(0), array(1), array(2)]}}
{'col1': {'original_col1': [array(3), array(4), array(5)], 'index': array([0, 1, 2])}, 'col2': {'copied_col1': [array(3), array(4), array(5)]}}
{'col1': {'original_col1': [array(6), array(7), array(8)], 'index': array([0, 1, 2])}, 'col2': {'copied_col1': [array(6), array(7), array(8)]}}
```

## 从数据处理管道中获取Python字典

直接迭代数据集对象即可获得字典类型的数据。当使用迭代器获取数据时，数据处理管道会尝试将字典对象中的所有值转成Tensor类型（如果`output_numpy`设置为`True`，则将转为NumPy类型）。

注意，上述类型转换操作是递归进行的，即应用于嵌套字典内的所有值以及列表和元组内的所有元素。无法被转成NumPy数组/Tensor类型的对象（例如类对象）会被直接传入到模型，若模型无法处理该对象类型将会报错。

下面的例子展示了通过迭代器获取字典数据或其他数据。

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

输出:

```python
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

在模型训练/推理场景，从数据管道获取字典类型数据时有以下注意事项。

- 在[数据下沉](https://mindspore.cn/tutorials/experts/zh-CN/r2.0/optimize/execution_opt.html?#%E6%95%B0%E6%8D%AE%E4%B8%8B%E6%B2%89)模式下，由于数据下沉通道当前无法支持字典类型的数据，字典类型的数据发送到下沉通道会造成错误。因此可以考虑关闭数据下沉模式（`dataset_sink_mode=False`），或在最后一个数据处理节点将字典类型的数据展开为列表或元组类型的数据，例如：

  ```python
  import numpy as np
  import mindspore.dataset as ds

  def my_generator():
      for i in range(5):
          col1 = {"my_data": np.array(i), "my_data2": np.array(i + 1)}
          yield col1,

  data = ds.GeneratorDataset(source=my_generator, column_names=["col1"])

  print(">>> get data in dict type")
  for d in data:
      print(d)

  def dict_to_tuple(d):
    return tuple([i for i in d.values()])

  # flatten the dict object bedfore it passed into network
  data = data.map(dict_to_tuple, input_columns=["col1"], output_columns=["my_data", "my_data2"])

  print(">>> get data in sequence type")
  for d in data:
      print(d)
  ```

- 在非数据下沉模式下，此特性没有使用限制，只需注意字典中存储的类型是否能够被模型识别和处理。