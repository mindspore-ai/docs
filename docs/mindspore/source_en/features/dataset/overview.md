# Data Processing Overview

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/features/dataset/overview.md)

MindSpore Dataset provides two types of data processing capabilities: pipeline mode and lightweight mode.

1. Pipeline mode: provides the concurrent data processing pipeline capability based on C++ Runtime. Users can define processes such as dataset loading, data transforms, and data batch process to implement efficient dataset loading, processing, and batching. In addition, the concurrency and cache can be adjusted to provide training data with zero Bottle Neck for NPU card training.

2. Lightweight mode: Users can perform data transform operations (e.g. Resize, Crop, HWC2CHW, etc.). Data processing of a single sample is performed.

## Pipeline Mode

Dataset pipeline defined by an API is used. After a training process is run, the dataset cyclically loads data from the dataset, processes data, and batch data, and then iterators for training.

![MindSpore Dataset Pipeline](https://www.mindspore.cn/docs/en/master/_images/dataset_pipeline_en.png)

As shown in the above figure, the mindspore dataset module makes it easy for users to define data preprocessing pipelines and transform samples in the dataset in the most efficient (multi-process / multi-thread) manner. The specific steps are as follows:

- Dataset loading: Users can easily load supported datasets using the Dataset class([Standard-format Dataset](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.loading.html#standard-format), [Vision Dataset](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.loading.html#vision), [NLP Dataset](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.loading.html#text), [Audio Dataset](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.loading.html#audio)), or load Python layer customized datasets through UDF Loader + [GeneratorDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset). At the same time, the loading class method can accept a variety of parameters such as sampler, data slicing, and data shuffle;

- Dataset operation: The user uses the dataset object method [.shuffle](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/mindspore.dataset.Dataset.shuffle.html#mindspore.dataset.Dataset.shuffle) / [.filter](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/mindspore.dataset.Dataset.filter.html#mindspore.dataset.Dataset.filter) / [.skip](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/mindspore.dataset.Dataset.skip.html#mindspore.dataset.Dataset.skip) / [.split](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/mindspore.dataset.Dataset.split.html#mindspore.dataset.Dataset.split) / [.take](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/operation/mindspore.dataset.Dataset.take.html#mindspore.dataset.Dataset.take) / … to further shuffle, filter, skip, and obtain the maximum number of samples of datasets;

- Dataset sample transform operation: The user can add data transform operations ([vision transform](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.vision), [nlp transform](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.text), [audio transform](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.audio)) to the map operation to perform transforms. During data preprocessing, multiple map operations can be defined to perform different transform operations to different fields. The data transform operation can also be a user-defined transform pyfunc (Python function);

- Batch: After the transforms of the samples, the user can use the [.batch](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/batch/mindspore.dataset.Dataset.batch.html#mindspore.dataset.Dataset.batch) operation to organize multiple samples into batches, or use self-defined batch logic with the parameter per_batch_map applied;

- Iterator: Finally, the user can use the dataset object method [.create_dict_iterator](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/iterator/mindspore.dataset.Dataset.create_dict_iterator.html#mindspore.dataset.Dataset.create_dict_iterator) or [.create_tuple_iterator](https://www.mindspore.cn/docs/en/master/api_python/dataset/dataset_method/iterator/mindspore.dataset.Dataset.create_tuple_iterator.html#mindspore.dataset.Dataset.create_tuple_iterator) to create an iterator, which can output the preprocessed data cyclically.

### Dataset Loading

The following describes common dataset loading methods, such as single dataset loading, dataset combination, dataset segmentation, and dataset saving.

#### Loading A Single Dataset

The dataset loading class is used to load training datasets from local disks, OBS, and shared storage to the memory. The dataset loading interface is as follows:

| Dataset API Category | API List  | Description |
|---|---|---|
| Standard-format Datasets | [MindDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.MindDataset.html#mindspore.dataset.MindDataset), [TFRecordDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.TFRecordDataset.html#mindspore.dataset.TFRecordDataset), [CSVDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.CSVDataset.html#mindspore.dataset.CSVDataset), etc. | MindDataset depends on the MindRecord format. For details, see [Format Conversion](https://www.mindspore.cn/tutorials/en/master/dataset/record.html) |
| Customized Datasets | [GeneratorDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.GeneratorDataset.html#mindspore.dataset.GeneratorDataset), [RandomDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.RandomDataset.html#mindspore.dataset.RandomDataset), etc. | GeneratorDataset loads user-defined DataLoaders. For details, see [Custom DataSets](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html#customizing-dataset) |
| Common Datasets | [ImageFolderDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.ImageFolderDataset.html#mindspore.dataset.ImageFolderDataset), [Cifar10Dataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.Cifar10Dataset.html#mindspore.dataset.Cifar10Dataset), [IWSLT2017Dataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.IWSLT2017Dataset.html#mindspore.dataset.IWSLT2017Dataset), [LJSpeechDataset](https://www.mindspore.cn/docs/en/master/api_python/dataset/mindspore.dataset.LJSpeechDataset.html#mindspore.dataset.LJSpeechDataset), etc. | Used for commonly used open source datasets |

You can configure different parameters for loading [datasets](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.loading.html#vision) to achieve different loading effects. Common parameters are as follows:

- `columns_list`: filters specified columns from the dataset. The parameter applies only to some dataset interfaces. The default value is None, indicating that all data columns are loaded.

- `num_parallel_workers`: configures the number of read concurrency for the dataset. The default value is 8.

- You can configure the sampling logic of the dataset by using the following parameters:

    - `shuffle`: specifies whether to enable shuffle. The default value is True.

    - `num_shards` and `shard_id`: specifies whether to shard a dataset. The default value is None, indicating that the dataset is not sharded.

    - For more sampling logic, see [Data Sampling](https://www.mindspore.cn/tutorials/en/master/dataset/sampler.html).

#### Dataset Combination

Dataset combination can combine multiple datasets in series/parallel mode to form a new dataset object, see [Data Operation](https://www.mindspore.cn/tutorials/en/master/dataset/eager.html#data-operation).

#### Dataset Segmentation

The dataset is divided into a training dataset and a validation dataset, which are used in a training process and a validation process, respectively, see [Data Operation](https://www.mindspore.cn/tutorials/en/master/dataset/eager.html#data-operation).

#### Dataset Saving

Re-save the dataset to the MindRecord data format, see [Data Operation](https://www.mindspore.cn/tutorials/en/master/dataset/eager.html#data-operation).

### Data Transforms

#### Common Data Transforms

Users can use a variety of data transformation operations:

- `.map(...)` operation: transform samples.
- `.filter(...)` operation: filter samples.
- `.project(...)` operation: sort and filter multiple columns.
- `.rename(...)` operation: rename a specified column.
- `.shuffle(...)` operation: shuffle data based on the buffer size.
- `.skip(...)` operation: skip the first n samples of the dataset.
- `.take(...)` operation: read only the first n samples of the dataset.

The following describes how to use the `.map(...)`.

- Use the data transform operation provided by Dataset in `.map(...)`

    Dataset provides a rich list of built-in [data transform operations](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.transforms.html#) that can be used directly in `.map(...)`. For details, see the [Map Transform Operation](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html#built-in-transforms).

- Use custom data transform operations in `.map(...)`

    Dataset also supports user-defined data transform operations. You only need to pass user-defined functions to `.map(...)` to return. For details, see [Customizing Map Transform Operations](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html#user-defined-transforms).

- Return the Dict data structure in `.map(...)`

    The dataset also supports the return of the Dict data structure in the user-defined data transform operation, which makes the defined data transform more flexible. For details, see [Custom Map Transform Operation Processing Dictionary Object](https://www.mindspore.cn/tutorials/en/master/dataset/python_objects.html#processing-dict-with-map-operation).

#### Automatic Augmentation

In addition to the preceding common data transform, the dataset also provides an automatic data transform mode, which can automatically perform data transform processing on an image based on a specific policy. For details, see [Automatic Augmentation](https://www.mindspore.cn/tutorials/en/master/dataset/augment.html).

### Data Batch

Dataset provides the `.batch(...)` operation, which can easily organize samples after data transform into batches. There are two methods:

1. The default `.batch(...)` operation organizes batch_size samples into data whose shape is (batch_size, ...). For details, see the [Batch Operation](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html#batch-dataset).

2. The customized `.batch(..., per_batch_map, ...)` operation allows users to organize multiple [np.ndarray, nd.ndarray, ...] data records in batches based on the customized logic. For details, see [Customizing Batch Operation](https://www.mindspore.cn/tutorials/en/master/dataset/python_objects.html#processing-dict-with-batch-operation).

### Dataset Iterator

After defining the dataset loading `(xxDataset) -> data processing (.map) -> data batch (.batch)` dataset pipeline, you can use the iterator method `.create_dict_iterator(...)` / `.create_tuple_iterator(...)` to output data. For details, see [Dataset Iteration](https://www.mindspore.cn/tutorials/en/master/beginner/dataset.html#iterating-a-dataset).

### Performance Optimization

#### Data Processing Performance Optimization

If the performance of the data processing pipeline is insufficient, you can further optimize the performance by referring to [Data Processing Performance Optimization](https://www.mindspore.cn/tutorials/en/master/dataset/optimize.html) to meet end-to-end training performance requirements.

#### Single-node Data Cache

In addition, in the inference scenario, to achieve ultimate performance, you can use the [Single-node Data Cache](https://www.mindspore.cn/tutorials/en/master/dataset/cache.html) to cache datasets in the local memory to accelerate dataset reading and preprocessing.

## Lightweight Mode

You can directly use the data transform operation to process a piece of data. The return value is the data transform result.

Data transform operations ([vision transform](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.vision), [nlp transform](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.text), [audio transform](https://www.mindspore.cn/docs/en/master/api_python/mindspore.dataset.transforms.html#module-mindspore.dataset.audio)) can be used directly like calling a common function. Common usage is: first initialize the data transformation object, then call the data transformation operation method, pass in the data to be processed, and finally get the result of the process. For more examples, see [Lightweight Data Transformation](https://www.mindspore.cn/tutorials/en/master/dataset/eager.html#lightweight-data-transformation).

## Other Feature

### Supporting Python Objects in Dataset Pipeline

Dataset pipeline accepts any Python type as input for some operations(such as user-defined dataset `GeneratorDataset`, user-defined `map` augmentation operation, `batch(per_batch_map=...)`. See [Supporting Python Objects in Dataset Pipeline](https://www.mindspore.cn/tutorials/en/master/dataset/python_objects.html).
