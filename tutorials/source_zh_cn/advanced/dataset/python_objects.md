<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced/dataset/python_objects.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

# 数据集管道支持Python对象

数据集管道中的特定操作（如自定义数据集`GeneratorDataset`、自定义`map`增强操作、自定义`batch(per_batch_map=...)`支持Python字典类型的输入。与其他类型相比的主要区别是：Python `dict`类型不会被转换成C++中的等价类型，而是以引用的形式保留在管道中。

注意，虽然目前只支持`dict`类型的对象传入管道，但您也可以将其他Python类型封装进`dict`中并传入到数据集管道中。

## 处理Python字典对象

### 输入Python字典到数据集管道

将`dict`输入到数据集管道中可在以下几个操作中实现：

1. 自定义数据集`GeneratorDataset`，用户将组织好的`dict`以返回值的形式输入到数据集管道中；
2. 自定义`map`增强操作，用户可以定义PyFunc，在该PyFunc中返回`dict`数据；
3. 自定义`batch(per_batch_map=...)`操作，用户在`per_batch_map`中处理并返回`dict`数据。

### 从数据集管道中获取Python字典

当数据被发送到模型（或使用独立的迭代器）时，数据集管道会尝试将字典对象中的所有值转成NumPy数组（并且如果`output_numpy`设置为`False`，则后续将转成`Tensor`）。这意味着所有数值和字符串类型的值都会转换成数组。

注意，这一步递归应用于嵌套字典内的所有值以及列表和元组内的所有元素。

其他无法被转成NumPy数组的类型（例如类对象）会被直接传入到模型。

> 该特性目前只支持在非数据下沉模式下使用（即`dataset_sink_mode`设置为`False`）。

## GeneratorDataset

一个字典被视为一列，字典中的元素没有限制（包括数量和键/值的类型）。
下面这个例子展示了如何通过`GeneratorDataset`将`dict`对象传送到数据集管道：

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

输出:

```python
{'col1': {'number': array(0), 'square': array(0)}, 'col2': array(0, dtype=int64)}
{'col1': {'number': array(1), 'square': array(1)}, 'col2': array(1, dtype=int64)}
{'col1': {'number': array(2), 'square': array(4)}, 'col2': array(2, dtype=int64)}
{'col1': {'number': array(3), 'square': array(9)}, 'col2': array(3, dtype=int64)}
{'col1': {'number': array(4), 'square': array(16)}, 'col2': array(4, dtype=int64)}
```

## Map操作

与`GeneratorDataset`相同，每个字典对象被看作一列，且其中的元素没有限制。

> 除了用户自定义函数以外，现有的数据集管道变换操作(`mindspore.dataset.transforms`)均不支持`dict`类型的输入。

这个例子说明如何通过`map`操作和PyFunc，将`dict`类型加入到数据集管道中：

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

## Batch操作

当对数据集使用`batch`操作时，如果有一列中有`dict`对象，数据集管道会将字典中的相对应的值组合在一起。因此，所有的`dict`对象都必须具有相同的键。

如果用户提供了`per_batch_map`函数，字典中的对应元素将根据键分组到Python列表中。

如果用户没有提供`per_batch_map`函数，字典中的相应元素将根据键被分组并转换成一个NumPy数组。`batch`操作的结果（对于该列）将是一个字典，其中所有值都是NumPy数组。如果这种转换产生了`np.object_`类型的数组，由于模型训练侧的限制，将向用户显示一条错误消息并且终止数据集管道。

此外，如果`per_batch_map`有一列返回一个`dict`，那么它的其他列都必须同样返回`dict`。

这个例子说明如何通过`batch`操作和`per_batch_map`函数，将`dict`类型加入到数据集管道中：

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

输出:

```python
{'col1': {'original_col1': [array(0), array(1), array(2), array(3), array(4)], 'index': array([0, 1, 2, 3, 4])}}
{'col1': {'original_col1': [array(5), array(6), array(7), array(8), array(9)], 'index': array([0, 1, 2, 3, 4])}}
{'col1': {'original_col1': [array(10), array(11), array(12), array(13), array(14)], 'index': array([0, 1, 2, 3, 4])}}
```

下面用一个综合的示例，展示如何对Python字典对象使用上述的各种操作。注意，该例中没有提供`per_batch_map`，因此在"powers"键中返回单个NumPy数组。

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

输出:

```python
{'col1': {'powers': array([[ 0,  0,  0],
       [ 1,  1,  1],
       [ 2,  4,  8],
       [ 3,  9, 27],
       [ 4, 16, 64]])}, 'col2': array([0, 1, 2, 3, 4], dtype=int64)}
```
