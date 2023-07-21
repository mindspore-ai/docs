# 数据处理常见问题与分析方法

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.11/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.11/tutorials/experts/source_zh_cn/debug/minddata_debug.md)&nbsp;&nbsp;

## 数据准备

数据准备阶段可能存在的问题有数据集路径问题以及MindRecord 文件读写问题，包括数据读取路径和保存路径问题、文件读写问题等。

### 数据集路径有中文

错误日志：

```python
RuntimeError: Unexpected error. Failed to open file, file path E:\深度学习\models-master\official\cv\ssd\MindRecord_COCO\test.mindrecord
```

解决方法有两种：

① 将 MindRecord 格式数据集的输出路径指定在纯英文路径下；

② MindSpore 升级到 1.6.0 之后版本。

参考实例链接：

[MindRecord 数据准备 - Unexpected error. Failed to open file_MindSpore](https://www.hiascend.com/forum/thread-0231107679243990127-1-1.html)

### MindRecord文件问题

#### 未删除重名文件

错误日志：

```python
MRMOpenError: [MRMOpenError]: MindRecord File could not open successfully.
```

参考解决方法：

① 代码中添加删除文件逻辑，保证每次保存文件前删除目录下的重名 MindRecord 文件。

② MindSpore 1.6.0 之后版本，定义`FileWriter`对象时，可以加上`overwrite=True`来实现覆盖写。

参考实例链接：

[MindSpore 数据准备 - MindRecord File could not open successfully](https://www.hiascend.com/forum/thread-0231107679243990127-1-1.html)

#### 文件被移动

错误日志：

```python
RuntimeError: Thread ID 1 Unexpected error. Fail to open ./data/cora
RuntimeError: Unexpected error. Invalid file, DB file can not match file
```

使用MindSpore 1.4及之前版本时，在Windows环境下, 生成MindRecord格式数据集文件后移动位置，文件不能被正常加载到MindSpore中使用。

参考解决方法：

① Windows 环境下生成的 MindRecord 格式文件不要移动位置。

② 将 MindSpore 升级到 1.5.0 以及之后版本，重新生成 MindRecord 格式数据集，即可正常拷贝移动。

参考实例链接：

[MindSpore 数据准备 - Invalid file,DB file can not match_MindSpore](https://www.hiascend.com/forum/thread-0229106992212728097-1-1.html)

#### 自定义数据时类型设置错误

错误日志：

```python
RuntimeError: Unexpected error. Invalid data, the number of schema should be positive but got: 0. Please check the input schema.
```

参考解决方法：

修改数据输入类型，使其与脚本中的类型定义保持一致。

参考实例链接：

[MindSpore 数据准备 - Unexpected error. Invalid data](https://www.hiascend.com/forum/thread-0231107678315400125-1-1.html)

## 数据加载

数据加载阶段可能存在的问题：资源配置问题、`GeneratorDataset`相关问题以及迭代器问题等。

### 资源配置问题

#### CPU核数设置问题

错误日志：

```python
RuntimeError: Thread ID 140706176251712 Unexpected error. GeneratorDataset’s num_workers=8, this value is not within the required range of [1, cpu_thread_cnt=2].
```

参考解决方法：

① 添加代码手动配置 CPU 核数：`ds.config.set_num_parallel_workers()`

② 使用更高版本的 MindSpore,目前的 MindSpore 1.6.0 版本会根据硬件中CPU的核数自适应配置，避免出现CPU核数过低导致报错。

参考实例链接：

[MindSpore 数据加载 - Unexpected error. GeneratorDataset's num_workers=8, this value is not within the required range of](https://bbs.huaweicloud.com/forum/thread-189861-1-1.html)

#### PageSize 设置问题

错误日志：

```python
RuntimeError: Syntax error. Invalid data, Page size: 1048576 is too small to save a blob row.
```

参考解决方法：

使用MindSpore的 set_page_size API，将 pagesize 设置大一点,设置方法如下：

```python
from mindspore.mindrecord import FileWriter
writer = FileWriter(file_name="test.mindrecord", shard_num=1)
writer.set_page_size(1 << 26) # 128MB
```

参考实例链接：

[MindSpore 数据加载 - Invalid data,Page size is too small"](https://www.hiascend.com/forum/thread-0231107680001698128-1-1.html)

### `GeneratorDataset` 相关问题

#### `GeneratorDataset` 线程卡死

无错误日志，线程卡死

在自定义的数据处理中，存在 ```numpy.ndarray, mindspore.Tensor```数据类型混用过程，并且错误地使用 `numpy.array(Tensor)`做转换，导致 GIL(Global Interpreter Lock) 锁得不到释放，`GeneratorDataset` 不能正常工作。

参考解决方法：

① 在定义`GeneratorDataset`的第一个入参 `source` 时，如果涉及调用 Python function，则使用`numpy.ndarray`数据类型。

② 使用 `Tensor.asnumpy()`方法将`Tensor`转成`numpy.ndarray`。

参考实例链接：

[MindSpore 数据加载 - GeneratorDataset 线程卡死](https://www.hiascend.com/forum/thread-0232106992052900089-1-1.html)

#### 自定义数据返回类型不正确

错误日志：

```python
Unexpected error. Invalid data type.
```

错误描述：

自定义的 `Dataset` 或 `map` 操作中返回的是一个dict类型数据等，不是 numpy array或numpy array组成的元组（tuple）。因为其他类型（dict、object等）不是一个可控的数据类型同时也不明确其中的数据存储方式，因此报出 `Invalid type` 的错误。

参考解决方法：

① 检查自定义的数据处理部分的数据返回类型，需要返回 numpy array。

② 检查自定义加载数据时，`__getitem__`函数的数据返回类型，需要返回 tuple，且 tuple 的元素是 numpy 类型。

参考实例链接：

[MindSpore 数据集加载 - Unexpected error. Invalid data type_MindSpore](https://www.hiascend.com/forum/thread-0231107678315400125-1-1.html)

#### 自定义采样器初始化错误

错误日志：

```python
AttributeError: 'IdentitySampler' object has no attribute 'child_sampler'
```

参考解决方法：

在自定义的采样器初始化方法'\_\_init\_\_()'中需要使用'super().\_\_init\_\_()'调用父类的构造函数。

参考实例链接：

[MindSpore 数据集加载 - 'IdentitySampler' has no attribute child_sampler](https://www.hiascend.com/forum/thread-0229107679386960150-1-1.html)

#### 重复定义访问方式

错误日志：

```python
For 'Tensor', the type of "input_data" should be one of ...
```

参考解决方法：

选择合适的数据输入：随机访问（`__getitem__`），顺序访问（iter，next）两者选其一即可。

参考实例链接：

[MindSpore 数据集加载 - the type of `input_data` should be one of](https://www.hiascend.com/forum/thread-0229107683010760153-1-1.html)

#### 自定义数据返回字段与定义数目不一致

错误日志：

```python
RuntimeError: Exception thrown from PyFunc. Invalid python function, the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in column_names
```

参考解决方法：

检查 `GeneratorDataset` 返回与定义的`columns`字段是否一致。

参考实例链接：

[MindSpore 数据集加载 - Exception thrown from PyFunc](https://www.hiascend.com/forum/thread-0232107680321371137-1-1.html)

#### 用户脚本问题

错误日志：

```python
TypeError：parse() missing 1 required positionnal argument：'self'
```

参考解决方法：

单步调试代码，检查脚本中的语法，看是否缺少'()'等。

参考实例链接：

[MindSpore 数据集加载 - parse() missing 1 required positional](https://bbs.huaweicloud.com/forum/thread-189950-1-1.html)

#### 自定义数据集使用了算子或Tensor操作

错误日志：

```python
RuntimeError: Exception thrown from PyFunc. RuntimeError: mindspore/ccsrc/pipeline/pynative/pynative_execute.cc:1116 GetOpOutput] : The pointer[cnode] is null.
```

错误描述：

在自定义数据集里面使用了算子或Tensor操作，而数据处理时采用多线程并行处理，但算子或Tensor操作并不支持多线程执行，因此报错。

参考解决方法：

用户自定义的 Pyfunc 中，在数据集中的`__getitem__` 中不使用 MindSpore的Tensor操作或算子，建议先把入参转为 Numpy 类型，再通过 Numpy 相关操作实现相关功能。

参考实例链接：

[MindSpore 数据集加载 - The pointer[cnode] is null](https://www.hiascend.com/forum/thread-0230106992306834091-1-1.html)

#### 迭代初始化错误导致下标越界

错误日志：

```python
list index out of range
```

参考解决方法：

移除非必要的`index`成员变量，或者在每次迭代前对`index`赋值为 0 进行复位操作。

参考实例链接：

[MindSpore 数据集加载 - list index out of range](https://www.hiascend.com/forum/thread-0232107679694236136-1-1.html)

#### 未进行迭代初始化

错误日志：

```python
Unable to fetch data from GeneratorDataset, try iterate the source function of GeneratorDataset or check value of num_epochs when create iterator.
```

未进行迭代初始化导致`len`和`iter`数量不一致

参考解决方法：

在 iter 中加入清零操作

参考实例链接：

[MindSpore 数据集加载 - Unable to fetch data from GeneratorDataset](https://bbs.huaweicloud.com/forum/thread-189895-1-1.html)

### 迭代器相关问题

#### 重复创建迭代器

错误日志：

```python
oserror: [errno 24] too many open files
```

错误描述：

重复调用`iter()`会重复创建迭代器，而 `GeneratorDataset` 加载数据集时默认为多进程加载，每次打开的句柄在主进程停止前得不到释放，导致打开句柄数一直在增长。

参考解决方法：

使用 MindSpore 提供的dict迭代器 `create_dict_iterator()`和 tuple 迭代器 `create_tuple_iterator()`。

参考实例链接：

[MindSpore 数据加载 - too many open files](https://www.hiascend.com/forum/thread-0231107678973789126-1-1.html)

#### 错误使用从迭代器中获取数据的方法

错误日志：

```python
'DictIterator' has no attribute 'get_next'
```

参考解决方法：

可通过以下两种方式从迭代器中获取下一条数据：

```python
item = next(ds_test.create_dict_iterator())

for item in ds_test.create_dict_iterator():
```

参考实例链接：

[MindSpore 数据集加载- 'DictIterator' has no attribute 'get_next'](https://www.hiascend.com/forum/thread-0230107679565465123-1-1.html)

## 数据增强

数据增强阶段是对读取的数据进行数据处理，MindSpore目前支持如数据清洗shuffle、数据分批batch、数据重复repeat、数据拼接concat等常用数据处理操作。该阶段可能存在的问题有：数据类型问题、接口参数类型问题、消费节点冲突问题、数据分批问题以及内存资源问题等。

### 自定义数据增强操作调用第三方库API时数据类型错误

错误日志：

```python
TypeError：Invalid object  with type'<class 'PIL.Image.Image'>' and value'<PIL.Image.Image image mode=RGB size=180x180 at 0xFFFF6132EA58>'.
```

参考解决方法：

检查自定义函数中用到的第三方库API的数据类型要求，将输入的数据类型转换为该API期望的数据类型。

参考实例链接：

[MindSpore 数据增强 - TypeError: Invalid with type](https://www.hiascend.com/forum/thread-0229107679078336149-1-1.html)

### 自定义数据增强操作参数类型错误

错误日志：

```python
Exception thrown from PyFunc. TypeError: args should be Numpy narray. Got <class 'tuple'>.
```

参考解决方法：

修改 `call` 的入参为个数（且类型为 numpy.ndarray），除 `self` 外入参个数需要与 `input_columns` 中的参数个数保持一致，忽略 `input_columns` 时默认为全部的数据列。

参考实例链接：

[MindSpore 数据增强 - args should be Numpy narray](https://www.hiascend.com/forum/thread-0230107678833189122-1-1.html)

### 数据集有两个消费节点发生冲突

错误日志：

```python
ValueError: The data pipeline is not a tree (i.e. one node has 2 consumers)
```

错误描述：

dataset 定义上发生了分支，导致 dataset 无法确定分叉的走向。

参考解决方法：

检查数据集名称，通常一直保持同一个数据集名称即可。

参考实例链接：

[MindSpore 数据增强 - The data pipeline is not a tree](https://www.hiascend.com/forum/thread-0230107678474985121-1-1.html)

### 数据 shape 不一致导致的 batch 操作问题

错误日志：

```python
RuntimeError: Unexpected error. Inconsistent batch shapes, batch operation expect same shape for each data row, but got inconsistent shape in column 0, expected shape for this column is:, got shape:
```

参考解决方法：

① 检查需要进行 batch 操作的数据 shape，不一致时放弃进行 batch 操作。

② 如果一定要对 shape 不一致的数据进行 batch 操作，需要整理数据集，通过 pad 补全等方式进行输入数据 shape 的统一。

参考实例链接：

[MindSpore 数据增强 - Unexpected error. Inconsistent batch](https://bbs.huaweicloud.com/forum/thread-190394-1-1.html)

### 数据增强操作占用内存高

错误描述：

MindSpore 进行数据增强过程中，如果内存不足，可能会自动退出。 MindSpore 1.7及以后版本在内存占用超过80%时会进行告警，用户在进行大数据训练时，需要注意内存占用率，防止内存占用过高导致直接退出。

参考实例链接：

[MindSpore 数据增强 - 内存不足，自动退出](https://www.hiascend.com/forum/thread-0230107679768460124-1-1.html)
