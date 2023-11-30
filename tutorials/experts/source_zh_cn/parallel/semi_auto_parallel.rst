半自动并行
========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/semi_auto_parallel.rst
    :alt: 查看源文件

.. toctree::
  :maxdepth: 1
  :hidden:

  operator_parallel
  optimizer_parallel
  pipeline_parallel

半自动并行支持多种并行模式的自动混合使用，包括：

- `算子级并行 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/operator_parallel.html>`_：算子级并行是指以算子为单位，把输入张量和模型参数切分到多台设备上进行计算，提升整体速度。
- `优化器并行 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/optimizer_parallel.html>`_：优化器并行可以减少多台设备对于相同权重更新的冗余计算，将计算量分散到多个设备上。
- `流水线并行 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/pipeline_parallel.html>`_：流水线并行是指将模型按层切分，每个设备只处理模型中某一部分。
