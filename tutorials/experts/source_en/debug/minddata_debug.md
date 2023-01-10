# Common Data Processing Errors and Analysis Methods

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/debug/minddata_debug.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Data Preparation

Common errors you may encounter in the data preparation phase include dataset path and MindRecord file errors when you read or save data from or to a path or when you read or write a MindRecord file.

### The Dataset Path Contains Chinese Characters

Error log:

```python
RuntimeError: Unexpected error. Failed to open file, file path E:\深度学习\models-master\official\cv\ssd\MsindRecord_COCO\test.mindrecord
```

Two solutions are available:

1. Specify the output path of the MindRecord dataset to a path containing only English characters.

2. Upgrade MindSpore to a version later than 1.6.0.

For details, visit the following website:

[MindRecord Data Preparation - Unexpected error. Failed to open file_MindSpore](https://bbs.huaweicloud.com/forum/thread-183183-1-1.html)

### MindRecord File Error

#### The Duplicate File Is Not Deleted

Error log:

```python
MRMOpenError: [MRMOpenError]: MindRecord File could not open successfully.
```

Solution:

1. Add the file deletion logic to the code to ensure that the MindRecord file with the same name in the directory is deleted before the file is saved.

2. In versions later than MindSpore 1.6.0, when defining the `FileWriter` object, add `overwrite=True` to implement overwriting.

For details, visit the following website:

[MindSpore Data Preparation - MindRecord File could not open successfully](https://bbs.huaweicloud.com/forum/thread-184006-1-1.html)

#### The File Is Moved

Error log:

```python
RuntimeError: Thread ID 1 Unexpected error. Fail to open ./data/cora
RuntimeError: Unexpected error. Invalid file, DB file can not match file
```

When MindSpore 1.4 or an earlier version is used, in the Windows environment, after a MindRecord dataset file is generated and moved, the file cannot be loaded to MindSpore.

Solution:

1. Do not move the MindRecord file generated in the Windows environment.

2. Upgrade MindSpore to 1.5.0 or a later version and regenerate a MindRecord dataset. Then, the dataset can be copied and moved properly.

For details, visit the following website:

[MindSpore Data Preparation - Invalid file,DB file can not match_MindSpore](https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=183187&page=1&authorid=&replytype=&extra=#pid1436140)

#### The User-defined Data Type Is Incorrect

Error log:

```python
RuntimeError: Unexpected error. Invalid data, the number of schema should be positive but got: 0. Please check the input schema.
```

Solution:

Modify the input data type to ensure that it is consistent with the type definition in the script.

For details, visit the following website:

[MindSpore Data Preparation - Unexpected error. Invalid data](https://bbs.huaweicloud.com/forum/thread-189349-1-1.html)

## Data Loading

In the data loading phase, errors may be reported in resource configuration, `GeneratorDataset`, and iterators.

### Resource Configuration

#### Incorrect Number of CPU Cores

Error log:

```python
RuntimeError: Thread ID 140706176251712 Unexpected error. GeneratorDataset's num_workers=8, this value is not within the required range of [1, cpu_thread_cnt=2].
```

Solution:

1. Add the following code to manually configure the number of CPU cores: `ds.config.set_num_parallel_workers()`

2. Upgrade to MindSpore 1.6.0, which automatically adapts to the number of CPU cores in the hardware to prevent errors caused by insufficient CPU cores.

For details, visit the following website:

[MindSpore Data Loading - Unexpected error. GeneratorDataset's num_workers=8, this value is not within the required range of](https://bbs.huaweicloud.com/forum/thread-189861-1-1.html)

#### Incorrect PageSize Setting

Error log:

```python
RuntimeError: Syntax error. Invalid data, Page size: 1048576 is too small to save a blob row.
```

Solution:

Call the set_page_size API to set pagesize to a larger value. The setting method is as follows:

```python
from mindspore.mindrecord import FileWriter
writer = FileWriter(file_name="test.mindrecord", shard_num=1)
writer.set_page_size(1 << 26) # 128MB
```

For details, visit the following website:

[MindSpore Data Loading - Invalid data,Page size is too small"](https://bbs.huaweicloud.com/forum/thread-190004-1-1.html)

### `GeneratorDataset`

#### Suspended `GeneratorDataset` Thread

No error log is generated, and the thread is suspended.

During customized data processing, the `numpy.ndarray` and `mindspore.Tensor` data type are mixed and the `numpy.array(Tensor)` type is incorrectly used for conversion. As a result, the global interpreter lock (GIL) cannot be released and the `GeneratorDataset` cannot work properly.

Solution:

1. When defining the first input parameter `source` of `GeneratorDataset`, use the `numpy.ndarray` data type if a Python function needs to be invoked.

2. Use the `Tensor.asnumpy()` method to convert `Tensor` to `numpy.ndarray`.

For details, visit the following website:

[MindSpore Data Loading - Suspended GeneratorDataset Thread](https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=183188&page=1&authorid=&replytype=&extra=#pid1436147)

#### Incorrect User-defined Return Type

Error log:

```python
Unexpected error. Invalid data type.
```

Error description:

A user-defined `Dataset` or `map` operation returns data of the dict type, not a numpy array or a tuple consisting of numpy arrays. Data types (such as dict and object) other than numpy array or a tuple consisting of numpy arrays are not controllable and the data storage mode is unclear. As a result, the `Invalid type` error is reported.

Solution:

1. Check the return type of the customized data processing. The return type must be numpy array or a tuple consisting of numpy arrays.

2. Check the return type of the `__getitem__` function during customized data loading. The return type must be a tuple consisting of numpy arrays.

For details, visit the following website:

[MindSpore Dataset Loading - Unexpected error. Invalid data type_MindSpore](https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=183190&page=1&authorid=&replytype=&extra=#pid1436154)

#### User-defined Sampler Initialization Error

Error log:

```python
AttributeError: 'IdentitySampler' object has no attribute 'child_sampler'
```

Solution:

In the user-defined sampler initialization method '\_\_init\_\_()', use 'super().\_\_init\_\_()' to invoke the constructor of the parent class.

For details, visit the following website:

[MindSpore Dataset Loading - 'IdentitySampler' has no attribute child_sampler](https://bbs.huaweicloud.com/forum/thread-184010-1-1.html#pid1439794)

#### Repeated Access Definition

Error log:

```python
For 'Tensor', the type of "input_data" should be one of ...
```

Solution:

Select a proper data input method: random access (`__getitem__`) or sequential access (iter, next).

For details, visit the following website:

[MindSpore Dataset Loading - the type of `input_data` should be one of](https://bbs.huaweicloud.com/forum/thread-184041-1-1.html)

#### Inconsistency Between the Fields Returned by the User-defined Data and the Defined Fields

Error log:

```python
RuntimeError: Exception thrown from PyFunc. Invalid python function, the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in column_names
```

Solution:

Check whether the fields returned by `GeneratorDataset` are the same as those defined in `columns`.

For details, visit the following website:

[MindSpore Dataset Loading -Exception thrown from PyFunc](https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=189645&page=1&authorid=&replytype=&extra=#pid1474252)

#### Incorrect User Script

Error log:

```python
TypeError: parse() missing 1 required positionnal argument: 'self'
```

Solution:

Debug the code step by step and check the syntax in the script to see whether '()' is missing.

For details, visit the following website:

[MindSpore Dataset Loading - parse() missing 1 required positional](https://bbs.huaweicloud.com/forum/thread-189950-1-1.html)

#### Incorrect Use of Tensor Operations or Operators in Custom Datasets

Error log:

```python
RuntimeError: Exception thrown from PyFunc. RuntimeError: mindspore/ccsrc/pipeline/pynative/pynative_execute.cc:1116 GetOpOutput] : The pointer[cnode] is null.
```

Error description:

Tensor operations or operators are used in custom datasets. Because data processing is performed in multi-thread parallel mode and tensor operations or operators do not support multi-thread parallel execution, an error is reported.

Solution:

In the user-defined Pyfunc, do not use MindSpore tensor operations or operators in `__getitem__` in the dataset. You are advised to convert the input parameters to the Numpy type and then perform Numpy operations to implement related functions.

For details, visit the following website:

[MindSpore Dataset Loading - The pointer[cnode] is null](https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=183191)

#### Index Out of Range Due to Incorrect Iteration Initialization

Error log:

```python
list index out of range
```

Solution:

Remove unnecessary `index` member variables, or set `index` to 0 before each iteration to perform the reset operation.

For details, visit the following website:

[MindSpore Dataset Loading - list index out of range](https://bbs.huaweicloud.com/forum/thread-184036-1-1.html)

#### No Iteration Initialization

Error log:

```python
Unable to fetch data from GeneratorDataset, try iterate the source function of GeneratorDataset or check value of num_epochs when create iterator.
```

The value of `len` is inconsistent with that of `iter` because iteration initialization is not performed.

Solution:

Clear the value of `iter`.

For details, visit the following website:

[MindSpore Dataset Loading - Unable to fetch data from GeneratorDataset](https://bbs.huaweicloud.com/forum/thread-189895-1-1.html)

### Iterator

#### Repeated Iterator Creation

Error log:

```python
oserror: [errno 24] too many open files
```

Error description:

If `iter()` is repeatedly called, iterators are repeatedly created. However, because `GeneratorDataset` loads datasets in multi-thread mode by default, the handles opened each time cannot be released before the main process stops. As a result, the number of opened handles keeps increasing.

Solution:

Use the dict iterator `create_dict_iterator()` and tuple iterator `create_tuple_iterator()` provided by MindSpore.

For details, visit the following website:

[MindSpore Data Loading - too many open files](https://bbs.huaweicloud.com/forum/thread-184134-1-1.html)

#### Improper Data Acquisition from the Iterator

Error log:

```python
'DictIterator' has no attribute 'get_next'
```

Solution:

You can obtain the next piece of data from the iterator in either of the following ways:

```python
item = next(ds_test.create_dict_iterator())

for item in ds_test.create_dict_iterator():
```

For details, visit the following website:

[MindSpore Dataset Loading - 'DictIterator' has no attribute 'get_next'](https://bbs.huaweicloud.com/forum/thread-184026-1-1.html#pid1439832)

## Data Augmentation

In the data augmentation phase, the read data is processed. Currently, MindSpore supports common data processing operations, such as shuffle, batch, repeat, and concat. You may encounter the following errors in this phase: data type errors, interface parameter type errors, consumption node conflict, data batch errors, and memory resource errors.

### Incorrect Data Type for Invoking A Third-party Library API in A User-defined Data Augmentation Operation

Error log:

```python
TypeError: Invalid object with type'<class 'PIL.Image.Image'>' and value'<PIL.Image.Image image mode=RGB size=180x180 at 0xFFFF6132EA58>'.
```

Solution:

Check the data type requirements of the third-party library API used in the user-defined function, and convert the input data type to the data type expected by the API.

For details, visit the following website:

[MindSpore Data Augmentation - TypeError: Invalid with type](https://bbs.huaweicloud.com/forum/thread-184123-1-1.html)

### Incorrect Parameter Type in A User-defined Data Augmentation Operation

Error log:

```python
Exception thrown from PyFunc. TypeError: args should be Numpy narray. Got <class 'tuple'>.
```

Solution:

Change the number of input parameters of `call` (except `self`) to the number of parameters in `input_columns` and their type to numpy.ndarray. If `input_columns` is ignored, the number of all data columns is used by default.

For details, visit the following website:

[MindSpore Data Augmentation - args should be Numpy narray](https://bbs.huaweicloud.com/forum/forum.php?mod=viewthread&tid=183196&page=1&authorid=&replytype=&extra=#pid1436178)

### Consumption Node Conflict in the Dataset

Error log:

```python
ValueError: The data pipeline is not a tree (i.e. one node has 2 consumers)
```

Error description:

A branch occurs in the dataset definition. As a result, the dataset cannot determine the direction.

Solution:

Check the dataset name. Generally, retain the same dataset name.

For details, visit the following website:

[MindSpore Data Augmentation - The data pipeline is not a tree](https://bbs.huaweicloud.com/forum/thread-183193-1-1.html)

### Improper Batch Operation Due to Inconsistent Data Shapes

Error log:

```python
RuntimeError: Unexpected error. Inconsistent batch shapes, batch operation expect same shape for each data row, but got inconsistent shape in column 0, expected shape for this column is:, got shape:
```

Solution:

1. Check the shapes of the data that requires the batch operation. If the shapes are inconsistent, cancel the batch operation.

2. If you need to perform the batch operation on the data with inconsistent shapes, sort out the dataset and unify the shapes of the input data by padding.

For details, visit the following website:

[MindSpore Data Augmentation - Unexpected error. Inconsistent batch](https://bbs.huaweicloud.com/forum/thread-190394-1-1.html)

### High Memory Usage Due to Data Augmentation

Error description:

If the memory is insufficient when MindSpore performs data augmentation, MindSpore may automatically exit. In MindSpore 1.7 and later versions, an alarm is generated when the memory usage exceeds 80%. When performing large-scale data training, pay attention to the memory usage to prevent direct exit due to high memory usage.

For details, visit the following website:

[MindSpore Data Augmentation - Automatic Exit Due to Insufficient Memory](https://bbs.huaweicloud.com/forum/thread-190001-1-1.html)
