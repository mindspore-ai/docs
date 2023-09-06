并行策略优化
========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/strategy_optimize.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  strategy_select
  split_technique
  multiple_copy

----------------------

并行策略优化包括：

- `策略选择 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/strategy_select.html>`_：根据模型规模和数据量大小，可以选择不同的并行策略，以提高训练效率和资源利用率。
- `切分技巧 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/split_technique.html>`_：切分技巧是指通过手动配置某些关键算子的切分策略，减少张量重排布来提升训练效率。
- `多副本 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/multiple_copy.html>`_：多副本是指在一个迭代步骤中，将一个训练batch拆分成多个，将模型并行通信与计算进行并发，提升资源利用率。
