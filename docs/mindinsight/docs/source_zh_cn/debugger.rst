调试器
==================================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3.q1/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/r2.3.q1/docs/mindinsight/docs/source_zh_cn/debugger.rst
    :alt: 查看源文件

MindSpore调试器是为图模式训练提供的调试工具，可以用来查看并分析计算图节点的中间结果。
在MindSpore图模式的训练过程中，用户无法方便地获取到计算图中间节点的结果，使得训练调试变得很困难。使用MindSpore调试器，用户可以：

- 在MindSpore Insight调试器界面结合计算图，查看图节点的输出结果；
- 设置监测点，监测训练异常情况（比如检查张量溢出），在异常发生时追踪错误原因；
- 查看权重等参数的变化情况。
- 查看图节点和源代码的对应关系。

调试器有在线和离线两种模式。在线调试器可以在训练过程中同步进行可视化分析，优势是操作简单，可以查看训练中每一步的结果，比较适合中小网络的调试，在训练网络过大时会导致内存不足。离线调试器是基于训练的Dump数据进行可视化分析，可以解决训练网络过大的情况下，在线调试器内存不足的问题，只需要事先保存的指定迭代的Dump数据，就能分析这些迭代。

.. toctree::
   :maxdepth: 1

   debugger_online
   debugger_offline
