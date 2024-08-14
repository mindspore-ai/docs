故障恢复
========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/recover.rst
    :alt: 查看源文件

.. toctree::
  :maxdepth: 1
  :hidden:

  disaster_recover
  fault_recover

在分布式并行训练过程中，遇到计算节点的故障或通信中断等问题，MindSpore有三种恢复方式：

- 模型重新加载：在训练时，通过配置参数合并保存，每张卡均保存了完整的模型参数文件，发生故障后可以直接加载之前保存的checkpoint进行恢复。详细请参考模型保存与加载中的 `模型加载 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/model_loading.html>`_ 。
- `动态组网场景下故障恢复 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/disaster_recover.html>`_：在动态组网启动场景下，若某个进程出现故障，其他进程会进入等待状态，可以通过重新拉起故障进程使得训练任务继续进行，而无需重启集群（目前仅支持GPU硬件平台）。
- `基于冗余信息的故障恢复 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/fault_recover.html>`_：在大模型训练中，根据数据并行的维度所划分的设备，他们的模型参数是相同的。根据这个原理，可以利用这些冗余的参数信息作为备份，在一个节点故障时，利用相同参数的另一节点就可以恢复故障的节点。
