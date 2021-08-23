class Cifar100Dataset(MappableDataset):
    """
    用于读取和解析Cifar100数据集的源数据集。

    生成的数据集有三列 :py:obj:`[image, coarse_label, fine_label]`。
    :py:obj:`image` 是uint8类型。
    :py:obj:`coarse_label` 和 :py:obj:`fine_labels` 都是uint32类型的标量。

    参数:
        dataset_dir (str): 包含数据集的根目录。
        usage (str, optional): 数据集的用途，可以是 `train`, `test` 或 `all`。 使用`train`参数将会读取50,000训练样本, `test` 将会读取10,000测试样本, `all` 将会读取全部60,000样本(默认值为None, 即全部样本图片)。
        num_samples (int, optional): 数据集包含的图片数量(默认值为None, 即全部样本图片)。
        num_parallel_workers (int, optional): 读取数据的线程数(默认值None, 在配置文件中进行配置）。
        shuffle (bool, optional): 是否对数据集进行shuffle操作(默认值None, 详情见下表参数及预期行为所示)。
        sampler (Sampler, optional): 用于从数据集中选择样本的对象(默认值None, 详情见下表参数及预期行为所示)。
        num_shards (int, optional): 数据集将被划分的份数(默认值None)。指定此参数后, `num_samples` 表示每份样本中的最大样本数。
        shard_id (int, optional): num_shards参数中每份的id (默认值None)。 只有当指定了num_shards才能指定此参数。
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
        -  此数据集可以采用sampler参数，`sampler`和`shuffle`是互斥的。
        下表展示了几种合法的输入参数及预期的行为。

    .. list-table:: 使用`sampler` 和`shuffle`参数样例及预期的行为
       :widths: 25 25 50
       :header-rows: 1

       * - 参数`sampler`
         - 参数`shuffle`
         - Expected Order Behavior
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
        >>> cifar100_dataset_dir = "/path/to/cifar100_dataset_directory"
        >>>
        >>> # 1)  依次获取CIFAR100 数据集中的所有样本
        >>> dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir, shuffle=False)
        >>>
        >>> # 2)  从CIFAR100数据集中随机抽取350个样本
        >>> dataset = ds.Cifar100Dataset(dataset_dir=cifar100_dataset_dir, num_samples=350, shuffle=True)
        >>>
        >>> #  提示： 在CIFAR100数据集中, 每个字典都有 "image", "fine_label" 和 "coarse_label"关键字

    关于CIFAR-100 数据集:
        | CIFAR-100 数据集和CIFAR-10 数据集非常相似, CIFAR-100有 100 个类别，每类包含600 张图片，其中 500 张训练图片and 100 测试图片。 这100个类别又被分成20个超类。 每个图片都有一个"精细" 分类标签（所属类别） 和 a "粗糙"类别标签 (所属超类)。

        | 以下为原始CIFAR-100 数据集结构。
        | 您可以将数据集解压成如下的文件结构，并通过Mindspore的API进行读取。
        | . 
        | ������ cifar-100-binary
        |      ������ train.bin
        |      ������ test.bin
        |      ������ fine_label_names.txt
        |      ������ coarse_label_names.txt