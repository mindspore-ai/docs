优化方法
========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/optimize_technique.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  strategy_optimize
  memory_optimize
  communication_optimize

----------------------

考虑到实际并行训练中，可能会对训练性能、吞吐量或规模有要求，可以从三个方面考虑优化：并行策略优化、内存优化和通信优化

- `并行策略优化 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/strategy_optimize.html>`_：并行策略优化主要包括并行策略的选择、算子级并行下的切分技巧以及多副本技巧。
- `内存优化 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/memory_optimize.html>`_：内存优化包括重计算、数据集切分、Host&Device异构和异构存储，主要目标是节省内存空间。
- `通信优化 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/communication_optimize.html>`_：通信优化包括通信融合和通信子图提取与复用，主要目标是减少通信延时，提升性能。
