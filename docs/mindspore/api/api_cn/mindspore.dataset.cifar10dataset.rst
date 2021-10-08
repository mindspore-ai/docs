mindspore.dataset.Cifar10Dataset
================================

.. py:class:: Cifar10Dataset(dataset_dir, usage=None, num_samples=None, num_parallel_workers=None, shuffle=None, sampler=None, num_shards=None, shard_id=None, cache=None)

    用于读取和解析Cifar10数据集，生成新的数据集文件。该API目前仅支持解析二进制版本的Cifar10文件。

    生成的数据集有两列: `[image, label]`。`image` 列的数据类型是uint8。`label` 列的数据是uint32类型的标量。

    **参数：**

        - **dataset_dir** (`str`): 包含数据集的根目录。
        - **usage** (`str, optional`): 加载的数据集分片，可以是 `train`, `test` 或 `all`。使用 `train` 参数将会读取50,000训练样本, `test` 将会读取10,000测试样本, `all` 将会读取全部60,000样本(默认值为None, 即全部样本图片)。
        - **num_samples** (`int, optional`): 指定从原始数据集读取的图片数量（可以小于数据集总数，默认值为None,即全部样本图片)。
        - **num_parallel_workers** (`int, optional`): 用于读取数据的线程数(默认值None,使用配置文件中的配置）
        - **shuffle** (`bool, optional`): 是否打乱数据集样本顺序(默认值None,详情见下表参数及预期行为所示)。
        - **sampler** (`Sampler, optional`): 用于加载数据集的采样器(默认值None,详情见下表参数及预期行为所示)。
        - **num_shards** (`int, optional`): 分布式训练时的数据集分片数(默认值None)。指定此参数后, `num_samples` 表示每份样本中的最大样本数。
        - **shard_id** (`int, optional`): 分布式训练时当前加载的分片ID(默认值None)。只有当指定了 `num_shards` 才能指定此参数。
        - **cache** (`DatasetCache, optional`): 单节点数据缓存，能够加快数据加载和处理的速度(默认值None, 即不使用缓存加速)。

    **异常：**

        - **RuntimeError:** 如果 `dataset_dir` 路径下不包含数据文件。
        - **RuntimeError:** 如果 `num_parallel_workers` 超过系统最大线程数。
        - **RuntimeError:** 如果同时设定 `sampler` 和 `shuffle` 参数。
        - **RuntimeError:** 如果同时设定 `sampler` 和 `sharding` 参数。
        - **RuntimeError:** 如果指定了 `num_shards` 参数，但是未指定 `shard_id` 参数。
        - **RuntimeError:** 如果指定了 `shard_id` 参数，但是未指定 `num_shards` 参数。
        - **ValueError:** 如果 `shard_id` 参数错误(小于0或者大于等于 `num_shards`)。

    .. note::

        此数据集可以指定sampler参数，`sampler` 和 `shuffle` 是互斥的。下表展示了几种合法的输入参数及预期的行为。

    .. list-table:: 使用 `sampler` 和 `shuffle` 参数样例及预期的行为。
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

            >>> cifar10_dataset_dir = "/path/to/cifar10_dataset_directory"
            >>>
            >>> # 1) 依次获取CIFAR10数据集中的所有样本
            >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, shuffle=False)
            >>>
            >>> # 2) 从CIFAR10数据集中随机抽取350个样本
            >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, num_samples=350, shuffle=True)
            >>>
            >>> # 3) 从CIFAR10数据集2分分布式训练样本中提取id为0的数据
            >>> dataset = ds.Cifar10Dataset(dataset_dir=cifar10_dataset_dir, num_shards=2, shard_id=0)
            >>>
            >>> # 提示：在CIFAR10数据集中,每个字典都有"image"和"label"关键字

    关于CIFAR-10数据集:

    CIFAR-10数据集由10类60000张32x32彩色图片组成，每类6000张图片。有50000个训练样本和10000个测试样本。图片分为飞机、汽车、鸟类、猫、鹿、狗、青蛙、马、船和卡车这10类。

    以下为原始CIFAR-10 数据集结构。您可以将数据集解压成如下的文件结构，并通过Mindspore的API进行读取。

    .. code-block::

        .
        └── cifar-10-batches-bin
            ├── data_batch_1.bin
            ├── data_batch_2.bin
            ├── data_batch_3.bin
            ├── data_batch_4.bin
            ├── data_batch_5.bin
            ├── test_batch.bin
            ├── readme.html
            └── batches.meta.text
