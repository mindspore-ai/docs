# 单节点缓存

`Linux` `Ascend` `GPU` `CPU` `中级` `高级`

<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [单节点缓存](#单节点缓存)
    - [概述](#概述)
    - [缓存基础使用](#缓存基础使用)
    - [缓存经过数据增强的数据](#缓存经过数据增强的数据)
    - [缓存共享](#缓存共享)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/source_zh_cn/advanced_use/cache.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

对于需要重复访问远程的数据集或从需要重复从磁盘中读取数据集的情况，可以使用单节点缓存算子将数据集缓存于本地内存中，以加速数据集的读取。

缓存算子依赖于在当前节点启动的缓存服务器，缓存服务器作为守护进程独立于用户的训练脚本而存在，主要用于提供缓存数据的管理，支持包括存储、查找、读取，以及发生缓存未命中时对于缓存数据的写入等操作。

若用户的内存空间不足以缓存所有数据集，则用户可以配置缓存算子使其将剩余数据缓存至磁盘。

##  缓存基础使用

- **Step 1:**

    在使用单节点缓存服务之前，首先需要启动缓存服务器：

    ```shell
    cache_admin --start
    ```

    **cache_admin命令支持以下参数：**
    - `-w`：设置缓存服务器的工作线程数量，默认情况下工作线程数量为32。
    - `-s`：设置若缓存数据的大小超过内存空间，则溢出至磁盘的数据文件路径，默认为`/tmp`路径。
    - `-h`：缓存服务器的ip地址，默认为127.0.0.1。
    - `-p`：缓存服务器的端口号。
    - `-g`： 生成一个缓存会话。
    - `-d`：删除一个缓存会话。
    - `-l`：设置日志等级。

- **Step 2:**

    随后，在Python训练脚本中使用`DatasetCache` API来定义一个名为`test_cache`的缓存实例：

    ```python
    import mindspore.dataset as ds
    import mindspore.common.dtype as mstype

    test_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)
    ```

    **DatasetCache支持以下参数：**
    - `session_id`： 缓存会话的id。
    - `size`：缓存最大内存空间占用，该参数以MB为单位，例如512GB的缓存空间应设置size=524288。
    - `spilling`：当内存空间超出所设置的最大内存空间占用时，是否允许将剩余的数据溢出至磁盘，默认为False。
    - `hostname`：连接至缓存服务器的ip地址，默认为127.0.0.1。
    - `port`：连接至缓存服务器的端口号。

    > - 在实际使用中，通常应当首先使用`cache_admin -g`命令从缓存服务器处获得一个缓存会话id并作为`session_id`参数，防止发生缓存会话冲突的状况。
    > - 设置`size=0`代表不限制缓存所使用的内存空间。使用此设置的用户需自行注意缓存的内存使用状况，防止因机器内存耗尽而导致缓存服务器进程被杀或机器重启的状况。
    > - 若设置`spilling=True`，则用户需确保所设置的磁盘路径具有写入权限以及足够的磁盘空间，以存储溢出至磁盘的缓存数据。
    > - 若设置`spilling=False`，则缓存服务器在耗尽所设置的内存空间后将不再写入新的数据。

- **Step 3:**

    最后，在创建数据集算子时将所创建的`test_cache`作为其`cache`参数传入：

    ```python
    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8, shape=[2])
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    # apply cache to dataset
    data = ds.RandomDataset(schema=schema, total_rows=4, num_parallel_workers=1, cache=test_cache)

    num_iter = 0
    for item in data.create_dict_iterator(num_epochs=1):  # each data is a dictionary
      # in this example, each dictionary has keys "image" and "label"
      print("{} image: {} label: {}".format(num_iter, item["image"], item["label"]))
      num_iter += 1
    ```

    ```
    0 image: [135 135] label: [59]
    1 image: [53 53] label: [146]
    2 image: [99 99] label: [27]
    3 image: [208 208] label: [169]
    ```

- **Step 4:**

    在训练结束后，可以选择将当前的缓存销毁并释放内存：

    ```shell
    # Destroy the session
    cache_admin –-destroy_session $session_id
    ```

    以上命令将销毁缓存会话id为`session_id`的缓存。

    若选择不销毁缓存，则该缓存会话中的缓存数据将继续存在，用户下次启动训练脚本时可以继续使用该缓存。

##  缓存经过数据增强的数据

缓存算子既支持对于原始数据集的缓存，也可以被应用于缓存经过数据增强处理后的数据。

直接缓存经过数据增强处理后的数据通常会带来更大的性能收益，因为被缓存的数据仅需要进行一次所需的数据增强处理，随后用户即可通过缓存直接获取经过增强处理后的数据。

- **Step 1:**

    同样，缓存经过数据增强处理的数据也需要首先启动缓存服务器：

    ```shell
    cache_admin --start
    ```

- **Step 2:**

    并在Python脚本中定义缓存实例：

    ```python
    import mindspore.dataset as ds
    import mindspore.common.dtype as mstype
    import mindspore.dataset.vision.c_transforms as c_vision

    test_cache = ds.DatasetCache(session_id=1, size=0, spilling=True)
    ```

- **Step 3:**

    最后，在创建用于数据增强的`Map`算子是将所创建的缓存实例传入：

    ```python
    schema = ds.Schema()
    schema.add_column('image', de_type=mstype.uint8, shape=[640, 480, 3])
    schema.add_column('label', de_type=mstype.uint8, shape=[1])

    data = ds.RandomDataset(schema=schema, total_rows=4, num_parallel_workers=1)

    # apply cache to map
    rescale_op = c_vision.Rescale(1.0 / 255.0, -1.0)
    data = data.map(operations=rescale_op, input_columns=["image"], cache=test_cache)

    num_iter = 0
    for item in data.create_dict_iterator(num_epochs=1):  # each data is a dictionary
      # in this example, each dictionary has keys "image" and "label"
      print("{} image shape: {} label: {}".format(num_iter, item["image"].shape, item["label"]))
      num_iter += 1
    ```

    ```
    0 image shape: (640, 480, 3) label: [99]
    1 image shape: (640, 480, 3) label: [203]
    2 image shape: (640, 480, 3) label: [37]
    3 image shape: (640, 480, 3) label: [242]
    ```

- **Step 4:**

    在训练结束后，可以选择将当前的缓存销毁并释放内存：

    ```shell
    # Destroy the session
    cache_admin –-destroy_session $session_id
    ```

## 缓存共享

对于分布式训练的场景，缓存算子还允许多个相同的训练脚本共享同一个缓存，共同从缓存中读写数据。

- **Step 1:**

    首先启动缓存服务器：

    ```shell
    cache_admin --start
    ```

- **Step 2:**

    在启动训练脚本的shell脚本中，生成一个缓存会话id：

    ```shell
    #!/bin/bash
    # This shell script will launch parallel pipelines

    # generate a session id that these parallel pipelines can share
    result=$(cache_admin -g 2>&1)
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "some error"
      exit 1
    fi

    # grab the session id from the result string
    session_id=$(echo $result | awk ‘{print $NF}’)
    ```

- **Step 3:**

    在启动训练脚本时将`session_id`以及其他参数传入：

    ```shell
    # make the session_id available to the python scripts
    num_devices=4

    for p in $(seq 0 $((${num_devices}-1))); do
     python my_training_script.py -–num_devices “$num_devices” –-device “$p” –-session_id $session_id &
    done
    ```

- **Step 4:**

    在python脚本内部接收传入的`session_id`，并在定义缓存实例时将其作为参数传入：

    ```python
    import mindspore.dataset as msds
    import mindspore.dataset.engine as de

    parser.add_argument('--session_id', type=int, default=1, help='Device num.')

    # use the session id passed in from the outside script when defining the cache
    test_cache = msds.DatasetCache(session_id = session_id, size = 0, spilling=False)
    ds = de.ImageFolderDataset(data_dir, num_samples=num_samples, cache = test_cache)
    ```

- **Step 5:**

    在训练结束后，可以选择将当前的缓存销毁并释放内存：

    ```shell
    # Destroy the session
    cache_admin –-destroy_session $session_id
    ```
