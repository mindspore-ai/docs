MindPandas文档
==============================

MindPandas是一个以分布式运行框架和多线程为底座，提供兼容Pandas API的高性能大数据处理工具包。MindPandas的架构图如下图所示：

.. raw:: html

    <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindpandas/docs/source_zh_cn/images/mindpandas_framework.png" width="700px" alt="" >

1. 底层的分布式运行框架或多进程后端提供分布式计算的能力，原始数据进行切分后，会调用此模块计算并返回给调用层。

2. 在Execution模块上层，将算子分为multiprocess operator、multithread operator，具体的map、reduce、fold等运算在此提供。

3. 在更上层的EagerFrame，作为MindPandas后端的通用数据格式，可依据数据的维度包装为最外层的DataFrame类或者Series类。

4. QueryCompiler层选择对应的运算逻辑，以在最外层API和EagerFrame之间进行数据处理和调用。最外层的API接口和Pandas接口参数一致。

设计特点
---------

1. MindPandas可以使用机器上的所有内核

   相较于原生Pandas的单线程实现，在任何给定时间只能使用一个CPU内核，MindPandas可以使用机器上的所有内核，或者整个集群的所有内核，使用如下所示：

   .. raw:: html

       <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindpandas/docs/source_zh_cn/images/mindpandas_multicore.png" width="700px" alt="" >

   MindPandas可以拓展到整个集群，利用整个集群的内存以及CPU资源，使用如下所示：

   .. raw:: html

       <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindpandas/docs/source_zh_cn/images/mindpandas_multimachine.png" width="700px" alt="" >

2. MindPandas与现有原生Pandas的API在接口使用上保持一致，设置MindPandas的后端运行模式即可运行脚本，在使用上只需将pandas的导入替换为：

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
API    pandas MindPandas
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

MindPandas初始版本包含以DataFrame、Series、Groupby和Other类共100个API。MindPandas后续版本将增加对更多API的支持，同时适配MindSpore训练等，敬请期待。

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

   mindpandas_DataFrame_api
   mindpandas_Series_api
   mindpandas_Groupby_api
   mindpandas_Other_api


.. toctree::
   :maxdepth: 1
   :caption: 参考文档

   faq

.. toctree::
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
