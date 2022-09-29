MindPandas文档
==============================

数据处理及分析是AI训练流程中重要的一环，其中表格数据类型是常见的数据表示形式。当前业界常用的数据分析框架Pandas提供了易用、丰富的接口，但由于其单线程的执行方式，在处理较大数据量时性能较差，同时因为其不支持分布式，导致无法处理超出单机内存的大数据量；另外，由于业界常用的数据分析框架与昇思MindSpore等AI框架是互相独立的，数据需要经过落盘、格式转换等步骤才能被训练，极大影响了使用效率。

MindPandas是一款兼容Pandas接口，同时提供分布式处理能力的数据分析框架，致力于提供支持大数据量、高性能的表格类型数据处理能力，同时又能与训练流程无缝结合，使得昇思MindSpore支持完整AI模型训练全流程的能力。

MindPandas的架构图如下图所示：

.. raw:: html

    <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindpandas/docs/source_zh_cn/images/mindpandas_architecture.png" width="700px" alt="" >

1. 最上层提供了与Pandas兼容的API，基于现有Pandas脚本，修改少量代码，即可切换到MindPandas进行分布式执行。

2. Distributed Query Compiler将API转换成为分布式的基础范式组合(map/reduce/injective_map等)，保证了后端逻辑的稳定性，当有新的算子实现时，可尝试转换为已有的通用计算范式组合。

3. Parallel Execution层提供了两种执行模式：多线程模式和多进程模式，用户可根据自己的实际场景进行选择。

4. MindPandas将原始数据进行切分，形成多个内部的Partition切片，随后每个Partition在不同的线程或进程同时执行相应的算子逻辑，从而实现数据的并行处理。

5. 最底层提供了插件化的算子执行逻辑，当前主要支持Pandas算子，后续会以插件的形式支持更多类型的算子逻辑。

设计特点
---------

1. MindPandas可以使用机器上的所有CPU核

   相较于原生Pandas的单线程实现，在任何给定时间只能使用一个CPU核，MindPandas可以使用机器上的所有CPU核，或者整个集群的所有CPU核，使用如下所示：

   .. raw:: html

       <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindpandas/docs/source_zh_cn/images/mindpandas_multicore.png" width="700px" alt="" >

   MindPandas可以拓展到整个集群，利用整个集群的内存以及CPU资源，使用如下所示：

   .. raw:: html

       <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindpandas/docs/source_zh_cn/images/cluster.png" width="700px" alt="" >

2. MindPandas与现有原生Pandas的API在接口使用上保持一致，设置MindPandas的后端运行模式即可运行脚本，在使用上只需将Pandas的导入替换为：

   .. code-block:: python

       # import pandas as pd
       import mindpandas as pd

   使用方便快捷，to_pandas接口还与现有Pandas代码的兼容。实现改动小，性能优的效果。

MindPandas性能介绍
------------------

MindPandas通过将原始数据分片，在分片的基础上进行分布式并行计算，以此大幅度减少计算时间。

以read_csv为例，使用8核CPU读取900MB大小的csv文件，结果如下所示：

测试场景：

- CPU：i7-8565u (4核8线程)
- 内存：16GB
- 数据大小：900MB csv文件

======== ====== ==========
API      pandas mindpandas
======== ====== ==========
read_csv 11.53s 5.62s
======== ====== ==========

.. code-block:: python

   import pandas as pd
   import mindpandas as mpd

   # pandas
   df = pd.read_csv("data.csv")

   # MindPandas
   mdf = mpd.read_csv("data.csv")

其他常用API如fillna，使用MindPandas均可获得数倍至数十倍不等的加速效果。

测试场景：

-  CPU：i7-8565u (4核8线程)
-  内存：16GB
-  数据大小：800MB (2,000,000行 \* 48列)

====== ====== ==========
API    pandas mindpandas
====== ====== ==========
fillna 0.77s  0.13s
====== ====== ==========

.. code:: python

   import pandas as pd
   import mindpandas as mpd

   df = df.fillna(1)

   # 可根据实际情况设置合适分片数
   mpd.set_partition_shape((4, 2))
   df = df.fillna(1)

常见的统计类API如max、min、sum、all、any等，在MindPandas中也通过并行化的方式大幅度提高了性能。

测试场景：

-  CPU：i7-8565u (4核8线程)
-  内存：16GB
-  数据大小：2GB (10,000,000行 \* 48列)

.. raw:: html

    <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindpandas/docs/source_zh_cn/images/performance_compare.png" width="700px" alt="" >

随着数据大小的增加，MindPandas的分布式并行处理所带来的优势会更明显，如下图所示在不同数据量下的性能对比：

.. raw:: html

    <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindpandas/docs/source_zh_cn/images/mindpandas_fillna.png" width="700px" alt="" >

注：mindpandas设为多进程模式，使用32核CPU

未来规划
---------

MindPandas初始版本包含DataFrame、Series、Groupby和Other类共100+API，后续将会增加对更多API的支持，以及数据的高效流转等功能，敬请期待。

使用MindPandas的典型场景
---------------------------------------

- MindPandas 数据处理

  提供与Pandas相同的接口，使用时替换引用的包即可进行分布式并行处理原生数据。

.. toctree::
   :maxdepth: 1
   :caption: 安装部署

   mindpandas_install

.. toctree::
   :maxdepth: 1
   :caption: 使用指南

   mindpandas_quick_start
   mindpandas_configuration

.. toctree::
   :maxdepth: 1
   :caption: API参考

   mindpandas.config
   mindpandas.DataFrame
   mindpandas.Series
   mindpandas.Groupby
   mindpandas.Others

.. toctree::
   :maxdepth: 1
   :caption: 参考文档

   faq

.. toctree::
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
