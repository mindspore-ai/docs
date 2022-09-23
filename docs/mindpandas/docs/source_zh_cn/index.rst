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

   相较于原生Pandas的单线程实现，在任何给定时间只能使用一个CPU内核，MindPandas可以使用机器上的所有内核，或者整个集群的所有内核，使用会类似如下：

   .. raw:: html

       <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindpandas/docs/source_zh_cn/images/mindpandas_multicore.png" width="700px" alt="" >

   额外的利用会提升性能，如果拓展到整个集群，MindPandas使用时如下图所示：

   .. raw:: html

       <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindpandas/docs/source_zh_cn/images/mindpandas_multimachine.png" width="700px" alt="" >

2. MindPandas与现有原生Pandas的API在接口使用上保持一致，设置MindPandas的后端运行模式即可运行脚本，在使用上只需将pandas的导入替换为：

   .. code-block:: python

       # import pandas as pd
       import mindpandas as pd

   使用方便快捷，to_pandas接口还与现有pandas代码的兼容。实现改动小，性能优化优的效果。

未来规划
---------

MindPandas初始版本包含以DataFrame、Series、Groupby和其他类共100个API。MindPandas后续版本将支持大数据集上的性能优化，对接MindSpore训练等内容，敬请期待。

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