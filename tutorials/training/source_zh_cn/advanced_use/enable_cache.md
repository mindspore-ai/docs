# 应用单节点数据缓存

`Linux` `Ascend` `GPU` `CPU` `数据准备` `中级` `高级`

<!-- TOC -->

- [应用单节点数据缓存](#应用单节点数据缓存)
    - [概述](#概述)
    - [配置环境](#配置环境)
    - [启动缓存服务器](#启动缓存服务器)
    - [创建缓存会话](#创建缓存会话)
    - [创建缓存实例](#创建缓存实例)
    - [插入缓存实例](#插入缓存实例)
    - [销毁缓存会话](#销毁缓存会话)
    - [关闭缓存服务器](#关闭缓存服务器)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/training/source_zh_cn/advanced_use/enable_cache.md" target="_blank"><img src="../_static/logo_source.png"></a>

## 概述

对于需要重复访问远程的数据集或需要重复从磁盘中读取数据集的情况，可以使用单节点缓存算子将数据集缓存于本地内存中，以加速数据集的读取。

下面，本教程将演示如何使用单节点缓存服务来缓存经过数据增强处理的数据。

## 配置环境

使用缓存服务前，需要安装MindSpore，并设置相关环境变量。以Conda环境为例，设置方法如下：

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path_to_conda}/envs/{your_env_name}/lib/python3.7/site-packages/mindspore:{path_to_conda}/envs/{your_env_name}/lib/python3.7/site-packages/mindspore/lib
export PATH=$PATH:{path_to_conda}/envs/{your_env_name}/bin
```

## 启动缓存服务器

在使用单节点缓存服务之前，首先需要启动缓存服务器：

```shell
$ cache_admin --start
Cache server startup completed successfully!
The cache server daemon has been created as process id 10394 and is listening on port 50052

Recommendation:
Since the server is detached into its own daemon process, monitor the server logs (under /tmp/mindspore/cache/log) for any issues that may happen after startup
```

若提示找不到`libpython3.7m.so.1.0`文件，尝试在虚拟环境下查找其路径并设置环境变量：

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{path_to_conda}/envs/{your_env_name}/lib
```

## 创建缓存会话

若缓存服务器中不存在缓存会话，则需要创建一个缓存会话，得到缓存会话id：

```shell
$ cache_admin -g
Session created for server on port 50052: 1493732251
```

缓存会话id由服务器随机分配。

## 创建缓存实例

在Python训练脚本中使用`DatasetCache` API来定义一个名为`some_cache`的缓存实例，并把上一步中创建的缓存会话id传入`session_id`参数：

```python
import mindspore.dataset as ds

some_cache = ds.DatasetCache(session_id=1493732251, size=0, spilling=True)
```

## 插入缓存实例

在应用数据增强算子时将所创建的`some_cache`作为其`cache`参数传入：

```python
import mindspore.common.dtype as mstype
import mindspore.dataset.vision.c_transforms as c_vision

schema = ds.Schema()
schema.add_column('image', de_type=mstype.uint8, shape=[640, 480, 3])

ds.config.set_seed(0)
ds.config.set_num_parallel_workers(1)

data = ds.RandomDataset(schema=schema, total_rows=4, num_parallel_workers=1)

# apply cache to map
rescale_op = c_vision.Rescale(1.0 / 255.0, -1.0)
data = data.map(input_columns=["image"], operations=rescale_op, cache=some_cache)

num_iter = 0
for item in data.create_dict_iterator(num_epochs=1):  # each data is a dictionary
    # in this example, each dictionary has a key "image"
    print("{} image shape: {}".format(num_iter, item["image"].shape))
    num_iter += 1
```

输出结果：

```text
0 image shape: (640, 480, 3)
1 image shape: (640, 480, 3)
2 image shape: (640, 480, 3)
3 image shape: (640, 480, 3)
```

通过`cache_admin --list_sessions`命令可以查看当前会话有四条数据，说明数据缓存成功。

```shell
$ cache_admin --list_sessions
Listing sessions for server on port 50052

     Session    Cache Id  Mem cached Disk cached  Avg cache size  Numa hit
  1493732251  3847471003           4         n/a         3686561         4
```

## 销毁缓存会话

在训练结束后，可以选择将当前的缓存销毁并释放内存：

```shell
$ cache_admin --destroy_session 1493732251
Drop session successfully for server on port 50052
```

以上命令将销毁缓存会话id为1493732251的缓存。

## 关闭缓存服务器

使用完毕后，可以选择关闭缓存服务器，该操作将销毁当前服务器中存在的所有缓存会话并释放内存。

```shell
$ cache_admin --stop
Cache server on port 50052 has been stopped successfully.
```
