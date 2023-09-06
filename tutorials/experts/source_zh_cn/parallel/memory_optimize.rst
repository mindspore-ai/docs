内存优化
========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/memory_optimize.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  distributed_gradient_accumulation
  recompute
  dataset_slice
  host_device_training
  memory_offload

----------------------

内存优化优化包括：

- `梯度累加 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/distributed_gradient_accumulation.html>`_：梯度累加通过在多个MicroBatch上计算梯度并将它们累加起来，然后一次性应用这个累积梯度来更新神经网络的参数。通过这种方法少量设备也能训练大Batch，有效减低内存峰值。
- `重计算 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/recompute.html>`_：重计算是一种以时间换空间的技术，通过不保存某些正向算子的计算结果，以节省内存空间，在计算反向算子时，需要用到正向结果再重新计算正向算子。
- `数据集切分 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/dataset_slice.html>`_：数据集单个数据过大甚至无法加载到单个设备的时候，可以对数据进行切分，进行分布式训练。数据集切分配合模型并行是有效降低显存占用的方式。
- `Host&Device异构 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/host_device_training.html>`_：在遇到参数量超过Device内存上限的时候，可以把一些内存占用量大且计算量少的算子放在Host端，这样能同时利用Host端内存大，Device端计算快的特性，提升了设备的利用率。
- `异构存储 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/memory_offload.html>`_：异构存储可以将暂时不需要用到的参数或中间结果拷贝到Host端内存或者硬盘，在需要时再恢复至Device端，从而减少显存占用。

