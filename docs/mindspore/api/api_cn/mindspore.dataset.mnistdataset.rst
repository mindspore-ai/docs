mindspore.dataset.MnistDataset
===============================

.. py:class:: MnistDataset(MappableDataset)

    用于读取和解析MNIST数据集,生成新的数据集文件。
    
    生成的数据集有两列: `[image, label]`。 `image` 列的数据类型为uint8。`label` 列的数据为uint32的标量。

    **参数：**

        - **dataset_dir** (`str`): 包含数据集的根目录。
        - **usage** (`str, optional`): 加载的数据集分片，可以是 `train`, `test` 或 `all`。使用 `train` 参数将会读取60,000训练样本, `test` 将会读取10,000测试样本, `all` 将会读取全部70,000样本(默认值为None, 即全部样本图片)。
        - **num_samples** (`int, optional`): 数据集包含的图片数量(默认值为None,即全部样本图片)。
        - **num_parallel_workers** (`int, optional`): 用于读取数据的线程数(默认值None,使用配置文件中的配置）。
        - **shuffle** (`bool, optional`): 是否打乱数据集样本顺序(默认值None,详情见下表参数及预期行为所示)。
        - **sampler** (`Sampler, optional`): 用于加载数据集的采样器(默认值None, 详情见下表参数及预期行为所示)。
        - **num_shards** (`int, optional`): 分布式训练时的数据集分片数(默认值None)。指定此参数后, `num_samples` 表示每份样本中的最大样本数。
        - **shard_id** (`int, optional`): 分布式训练时当前加载的分片ID(默认值None)。只有当指定了 `num_shards` 才能指定此参数。
        - **cache** (`DatasetCache, optional`): 单节点数据缓存，能够加快数据加载和处理的速度(默认值None, 即不使用缓存加速)。

    **异常：**

        - **RuntimeError:** 如果 `dataset_dir` 路径下不包含数据文件。
        - **RuntimeError:** 如果 `num_parallel_workers` 超过系统最大线程数
        - **RuntimeError:** 如果同时设定 `sampler` 和 `shuffle` 参数。
        - **RuntimeError:** 如果同时设定 `sampler` 和 `sharding` 参数。
        - **RuntimeError:** 如果指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError:** 如果指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError:** 如果 `shard_id` 参数错误(小于0或者大于等于 `num_shards`)。

    .. note::

        此数据集可以指定sampler参数，`sampler` 和 `shuffle` 是互斥的。下表展示了几种合法的输入参数及预期的行为。

    .. list-table:: 使用 `sampler` 和 `shuffle` 参数样例及预期的行为
       :widths: 25 25 50
       :header-rows: 1

       * - 参数 `sampler`
         - 参数 `shuffle`
         - 预期行为
       * - None
         - None
         - 随机顺序
       * - None
         - True
         - 随机顺序
       * - None
         - False
         - 原始顺序
       * - Sampler object
         - None
         - 由sampler定义的顺序
       * - Sampler object
         - True
         - 不合法
       * - Sampler object
         - False
         - 不合法

    **样例：**
    
        .. code-block::

            >>> mnist_dataset_dir = "/path/to/mnist_dataset_directory"
            >>>
            >>> #从MNIST数据集中读取3个样本
            >>> dataset = ds.MnistDataset(dataset_dir=mnist_dataset_dir, num_samples=3)
            >>>
            >>> # 提示：在MNIST数据集中,每个字典都有"image"和"label"关键字

    关于MNIST数据集:
    
    MNIST手写数字数据集是NIST数据集的子集，共有60,000个训练样本和10,000个测试样本。数字的大小已标准化，并居中。

    以下为原始MNIST数据集结构。您可以将数据集解压成如下的文件结构，并通过Mindspore的API进行读取。

    .. code-block::

        . 
        └── mnist_dataset_dir
            ├── t10k-images-idx3-ubyte
            ├── t10k-labels-idx1-ubyte
            ├── train-images-idx3-ubyte
            └── train-labels-idx1-ubyte
