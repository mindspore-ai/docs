# High Performance Data Processing Engine

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/design/data_engine.md)

## Background Introduction

The core of MindSpore training data processing engine is to efficiently and flexibly transform training samples (datasets) to a Tensor and provide that Tensor to the training network for training, with the following key features:

- Efficient data processing Pipeline, allowing data to flow within the Pipeline and achieving efficient processing capabilities.
- Provide a variety of data loading capabilities such as common datasets, specific format datasets (MindRecord), custom datasets, to meet a variety of dataset loading needs of users.
- Provide uniform sampling capabilities for multiple datasets, enabling flexible output of one copy of data.
- Provide a large number of C++ layer data processing operations, Python layer data processing operations, and support custom data processing operations, making it easy for users to use upon unpacking.
- Provide MindSpore dataset format (MindRecord), which facilitates users to convert their own datasets and then load them uniformly and efficiently through `MindDataset`.
- Provide an automatic data augmentation mode, and perform automatic data augmentation on images based on specific strategies.
- Provide single-node data caching capability to solve the problem of repeated loading and processing of data, reduce data processing overhead, and improve device-to-device training efficiency.

Please refer to the instructions for usage: [Data Loading And Processing](https://www.mindspore.cn/docs/en/r2.6.0rc1/features/dataset/overview.html)

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/docs/mindspore/source_en/design/images/data/data_engine_en.png)

MindSpore training data engine also provides efficient loading and sampling capabilities of datasets in fields, such as scientific computing-electromagnetic simulation, remote sensing large-format image processing, helping MindSpore achieve full-scene support.

## Data Processing Engine Design

### Design Goals and Ideas

The design of MindSpore considers the efficiency, flexibility and adaptability of data processing in different scenarios. The whole data processing subsystem is divided into the following modules:

![image](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/docs/mindspore/source_zh_cn/design/images/data/architecture.png)

- API: The data processing process is represented in MindSpore in the form of a graph, called a data graph. MindSpore provides Python API to define data graphs externally and implement graph optimization and graph execution internally.
- Data Processing Pipeline: Data loading and pre-processing multi-step parallel pipeline, which consists of the following components.

    - Adaptor: data graphs constructed in an upper-level language (e.g. Python) converted into a lower-level executable C++ data graph (Execution Tree).
    - Optimizer: Data graph optimizer to implement operations such as operator fusion and automatic parameter optimization.
    - Runtime: Running the execution engine of the optimized Execution tree.
    - Dataset Operations: A node in the Execution tree, corresponding to a specific step in the data processing pipeline, for example, the `ImageFolderDataset` and `MindDataset` operations for loading data from image folders, and the `map`, `shuffle`, `repeat`, `concat` and `split` operations for data processing.
    - Data Augmentation Operations: Also called Tensor operators, used to perform specific transformations on the Tensor, such as `Decode`, `Resize`, `Crop`, `Pad` operations, which are usually called by the `map` operation in Dataset Operations.

The results of the data augmentation are connected to the forward and backward computing system via a queue.

Based on the above design, the data processing engine implements the following Pipeline process:

In addition, due to the limited resources of device-side scenarios, MindSpore provides a set of more lightweight data processing Eager model, which can solve the problem that the data processing Pipeline of cloud-based scenarios is not applicable to the device-side. Users can directly perform data processing operations on a single image and then pass it into the model for inference.

### Ultimate Processing Performance

- Multi-stage data processing pipeline

    Unlike TensorFlow and PyTorch, MindSpore uses multi-stage parallel pipeline to build data processing Pipeline, which can plan the use of computational resources in a more fine-grained way. As shown above, each dataset operation contains an output Connector, which is an order-preserving buffer queue consisting of a set of blocking queues and counters. Each dataset operation takes cached data from the Connector of the upstream operation for processing, and then pushes this cache back to its own output Connector, and so on. The advantages of this mechanism include:

    - The dataset loading, `map`, `batch` and other operations are driven by a task scheduling mechanism. Tasks for each operation are independent of each other, and the contexts are linked through Connector.
    - Each operation can be implemented with fine-grained multi-threaded or multi-process parallel acceleration. Data framework provides the interfaces for users to adjust the number of threads of operation and control of multi-process processing, and users can flexibly control the processing speed of each node, and thus achieve optimal performance of the entire data processing Pipeline.
    - Support users to set the size of the Connector, which to a certain extent, can effectively control the memory utilization, and adapt to different network requirements for data processing performance.

    With this data processing mechanism, performing order-preserving on the output data is the key to ensure the training accuracy. Order preservation means that the data processing pipeline runs with the output data in the same order as it was before the data processing. MindSpore uses a round robin algorithm to ensure the orderliness of data during multi-threaded processing.

    The above figure is a data processing Pipeline, where the order-preserving operation occurs in takeout operation of the downstream `map` operation (4 concurrent) to take out the data in the upstream queue by single-threaded round robin. The Connector has two internal counters. `expect_consumer_` records how many `consumer`s have taken data from `queues_` and `pop_from_` records which internal blocking queue is about to perform the next takeout operation. `expect_consumer_` performs modulo operation on `consumer`, and `pop_from_` performs modulo operation on `producer`. When `expect_consumer_` is 0 again, it means that all the `local_queues_` have finished processing the previous batch of tasks and can continue to allocate and process the next batch of tasks, thus realizing multiple concurrent order-preserving processing from upstream to downstream `map` operations.

- Data processing and network computing pipeline

    The data processing pipeline continuously processes the data and sends the processed data to the Device-side cache, and after the execution of one Step, the data of the next Step is read directly from the Device's cache.

    - datat processing: for processing the dataset into the input needed by the network and passing it to the sending queue to ensure efficient data processing.

    - sending Queue: maintaining data queues to ensure that data processing and network computing processes do not interfere with each other, acting as a bridge.

    - network computing: get the data from the sending queue for iterative training.

    Each of the above three has its own role, independent of each other, to construct the entire training process Pipeline. Therefore, as long as the data queue is not empty, model training will not block due to waiting for training data.

- Cache technology

    When the dataset size is too large to be loaded all into the in-memory cache, some of the data used for training needs to be read from disk and may encounter I/O bottlenecks, increasing the hit ratio of cache in each Epoch is especially critical. Traditional cache management uses LRU strategy, which does not consider the read characteristics of deep learning data, i.e., data is read repeatedly between different Epochs, while it is read randomly in the same Epoch. Each piece of data has the same probability of being read, so it does not matter which piece of data is cached. Instead it is more critical that the data that has been cached is not swapped out before it is used. For this feature, we use a simple and efficient caching algorithm, i.e., once data is cached, it is not swapped out of the cache.

    During data graph optimization, MindSpore automatically generates caching operators based on the pipeline structure, caching both the original dataset and the results after data augmentation processing.

### Flexible Customization Capabilities

Users often have diverse needs for data processing, and processing logic that is not solidified in the framework needs to be supported by open customization capabilities. As a result, MindSpore provides flexible dataset loading methods, rich data processing operations, and mechanisms such as automatic data augmentation, dynamic Shape, and data processing Callback for developers to use in various scenarios.

- Flexible dataset loading methods

    To address the challenge of having a wide variety of datasets with different formats and organization, MindSpore provides three different methods of loading datasets:

    - For common datasets in each domain, they can be loaded directly by using MindSpore built-in API interface. MindSpore provides `CelebADataset`, `Cifar10Dataset`, `CocoDataset`, `ImageFolderDataset`, `MnistDataset`, `VOCDataset` and other common dataset loading interfaces to ensure performance while enabling users to use them out of the box.
    - For datasets that do not support direct loading at the moment, they can be converted to MindSpore data format, i.e. MindRecord, and then loaded through the `MindDataset` interface. MindRecord can normalize different dataset formats, with various advantages such as aggregated storage, efficient reading, fast coding and decoding, and flexible control of partition size.
    - Users can also write custom dataset reading classes in Python and then use the `GeneratorDataset` interface for dataset loading. This method allows for quick integration of existing code, but requires additional attention to data loading performance as it is a Python IO Reader.

- Support more operations by Python layer customization and C++ layer plug-in

    MindSpore has a rich set of built-in data processing operations, which can be divided into C++ layer and Python layer operations depending on the implementation. C++ layer operations tend to have better performance, while Python layer operations are easier to integrate with third-party libraries and are easier to implement. For the operation that the framework does not support, users can develop the C++ layer to implement the code, and the code is compiled in the form of a plug-in to register to MindSpore data processing Pipeline. The data processing logic is customized directly in the Python layer, and then called through the `map` operation.

- Support automatic data augmentation strategies

    MindSpore provides mechanisms for automatic image augmentation processing based on specific strategies, including probability-based automatic data augmentation and feedback-based automatic data augmentation, which can realize automatic selection and execution of operations to achieve improving training accuracy.

    For the ImageNet dataset, the automatic data augmentation strategy finally searched by the AutoAugment method contains 25 substrategy combinations. Each substrategy contains 2 transformations, and one substrategy combination is randomly selected for each image in the actual training. A certain probability is used to decide whether to execute each transformation in the substrategy. The flow is shown in the figure below.

    To support AutoAugment, an automatic data augmentation strategy, MindSpore provides the following interfaces.

    - RandomChoice, or random selection, allows the user to define a list of data augmentation operations, and the data processing process will select one data augmentation operation from the list with equal probability for each image.

        ```python
        from mindspore.dataset.transforms import RandomChoice
        from mindspore.dataset.vision import RandomCrop, RandomHorizontalFlip, RandomRotation

        transform_list = RandomChoice([RandomCrop((32, 32)),
                                       RandomHorizontalFlip(0.5),
                                       RandomRotation((90, 90))])
        ```

    - RandomApply, a random probability execution, allows the user to define a list of data augmentation operations and the corresponding probabilities, and the data augmentation operations in the list will be executed for each image with the specified probability, either all or none.

        ```python
        from mindspore.dataset.transforms import RandomApply
        from mindspore.dataset.vision import RandomCrop, RandomHorizontalFlip, RandomRotation

        transform_list = RandomApply([RandomCrop((32, 32)),
                                      RandomHorizontalFlip(0.5),
                                      RandomRotation((90, 90))], 0.8)
        ```

    - RandomSelectSubpolicy, a random subpolicy selection, allows users to define multiple lists of data augmentation operation subpolices and specify the probability of execution for each data augmentation operation in the subpolicy. During data processing, a subpolicy is first selected with equal probability for each image, and then whether each data augmentation operation is performed is decided in order according to the probability.

        ```python
        from mindspore.dataset.vision import RandomSelectSubpolicy, RandomRotation, RandomVerticalFlip, \
            RandomHorizontalFlip

        transform_list = RandomSelectSubpolicy([[(RandomRotation((45, 45)), 0.5),
                                                 (RandomVerticalFlip(), 1)],
                                                [(RandomRotation((90, 90)), 1),
                                                 (RandomHorizontalFlip(), 0.5)]])
        ```

    Automatic data augmentation operations can improve the training accuracy of the ImageNet dataset by about 1%.

- Support dynamic shape

    MindSpore supports custom control of the Shape of the output training data through `per_batch_map`, which satisfies that the network needs to adjust the data Shape based on different scenarios.

    - Users control the Shape of the generated data through user-defined functions (UDF), e.g. generate the data with the Shape (x, y, z, ...) at the nth Step.
    - The `per_batch_map` parameter of the `batch` operation is passed to this custom function to obtain the training data with different Shape.

- Callback mechanism makes data processing more flexible

    The function of dynamically adjusting the data augmentation logic according to the training results is implemented through the Callback mechanism, which provides more flexible automatic data augmentation.

    MindSpore supports users to implement their own data augmentation logic (User Defined Function, UDF) based on the DSCallback provided by data processing (including Epoch Start, Step Start, Step End, Epoch End) and add it to `map` operations to achieve more flexible data augmentation operations.

### Device-cloud Unified Architecture

- Unification of data and computational graphs

    MindIR is MindSpore functional IR based on graph representation, whose most central purpose is to serve automatic differential transformations. Automatic differentiation uses a transformation method based on a functional programming framework, so IR adopts a semantics close to that of ANF functional style.

    Typical scenarios for inference data graphs include size scaling, intermediate screenshots, normalization, and channel transformations.

    We save the inference data graphs as subgraphs in the generated model file (MindIR), so that the data processing process in the model can be loaded through a unified interface during inference to automatically perform data processing and get the input needed by the model, which simplifies user operations and improves ease of use.

- Lightweight data processing

    Data processing Pipeline occupies more system resources, including CPU and memory, during operation. Taking the training process of ImageNet as an example, the CPU usage reaches 20% and the memory usage reaches 30 to 50G. The resources that can be used are more abundant when training on the cloud side, but in device-side scenarios, this demand is often unacceptable. The initialization process of the data processing Pipeline is usually time-consuming, and also does not satisfy the device-side need for fast start-up and multiple training and inference. Therefore, MindSpore provides a set of data processing models that are lighter and more applicable to device-side scenarios, solving the problem that the data processing Pipeline for cloud-based scenarios is not applicable to the device-side.

    MindSpore supports independently use data processing single operation (Eager mode), supports various scenarios of inference, provides AI developers with greater flexibility based on Pipeline tuning architecture. At the same time, the Pipeline is lightened to achieve a light pipeline based on Pull Base, reducing resource consumption and improving processing speed.

With the above two methods, MindSpore ensures that a unified data processing architecture supports many different application scenarios.
