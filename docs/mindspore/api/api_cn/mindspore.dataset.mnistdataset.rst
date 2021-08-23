class MnistDataset(MappableDataset):
    """
    用于读取和解析MNIST数据集的源数据集。

    生成的数据集有两列: :py:obj:`[image, label]`.
    :py:obj:`image` 是uint8类型。
    :py:obj:`label` 是uint32类型的标量。

    参数:
        dataset_dir (str): 包含数据集的根目录。
        usage (str, optional): 数据集的用途，可以是`train`, `test` 或 `all` 。 使用`train`参数将会读取60,000训练样本, `test` 将会读取10,000测试样本, `all` 将会读取全部70,000样本(默认值为None, 即全部样本图片)。
        num_samples (int, optional): 数据集包含的图片数量(默认值为None, 即全部样本图片)。
        num_parallel_workers (int, optional): 读取数据的线程数(默认值None, 在配置文件中进行配置）。
        shuffle (bool, optional): 是否对数据集进行shuffle操作(默认值None, 详情见下表参数及预期行为所示)。
        sampler (Sampler, optional): 用于从数据集中选择样本(默认值None, 详情见下表参数及预期行为所示)。
        num_shards (int, optional): 数据集将被划分的份数(默认值None)。 指定此参数后, `num_samples` 表示每份样本中的最大样本数。
        shard_id (int, optional): num_shards参数中每份的id (默认值None)。只有当指定了num_shards才能指定此参数。
        cache (DatasetCache, optional): 使用张量缓存来加快数据处理速度。(默认值None, 即不使用缓存加速)。

    报错信息:
        RuntimeError: 如果dataset_dir参数中不包含数据文件。
        RuntimeError: 如果num_parallel_workers超过最大线程数。
        RuntimeError: 如果同时设定sampler和shuffle参数。
        RuntimeError: 如果同时设定sampler和sharding参数。
        RuntimeError: 如果指定了num_shards 参数，但是未指定shard_id参数。
        RuntimeError: 如果指定了shard_id参数， 但是未指定num_shards参数。
        ValueError: 如果shard_id 参数错误(< 0 或者 >= num_shards)。

    提示:
        - 此数据集可以采用sampler参数，`sampler`和`shuffle`是互斥的。
        下表展示了几种合法的输入参数及预期的行为。

    .. list-table:: 使用`sampler` 和`shuffle`参数样例及预期的行为
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
         - 顺序
       * - Sampler object
         - None
         - 由sampler采样器定义的顺序
       * - Sampler object
         - True
         - 不合法
       * - Sampler object
         - False
         - 不合法

    举例:
        >>> mnist_dataset_dir = "/path/to/mnist_dataset_directory"
        >>>
        >>> #从MNIST数据集中读取3个样本
        >>> dataset = ds.MnistDataset(dataset_dir=mnist_dataset_dir, num_samples=3)
        >>>
        >>> # 提示： 在CIFAR10数据集中, 每个字典都有 "image" 和 "label"关键字

    关于MNIST数据集:
    
        | MNIST手写数字数据集有60,000个训练样本和10,000个测试样本。它是NIST数据集的子集。数字大小已被标准化，并居中。

        | 以下为原始CIFAR-10 数据集结构。
        | 您可以将数据集解压成如下的文件结构，并通过Mindspore的API进行读取。
        | .
        | ������ mnist_dataset_dir
        |      ������ t10k-images-idx3-ubyte
        |      ������ t10k-labels-idx1-ubyte
        |      ������ train-images-idx3-ubyte
        |      ������ train-labels-idx1-ubyte
