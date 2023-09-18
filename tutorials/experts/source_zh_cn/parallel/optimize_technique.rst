优化方法
========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg
    :target: https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/parallel/optimize_technique.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  strategy_select
  split_technique
  multiple_copy
  distributed_gradient_accumulation
  recompute
  dataset_slice
  host_device_training
  memory_offload
  comm_fusion
  comm_subgraph

考虑到实际并行训练中，可能会对训练性能、吞吐量或规模有要求，可以从三个方面考虑优化：并行策略优化、内存优化和通信优化

- 并行策略优化：并行策略优化主要包括并行策略的选择、算子级并行下的切分技巧以及多副本技巧。
  
  - `策略选择 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/strategy_select.html>`_：根据模型规模和数据量大小，可以选择不同的并行策略，以提高训练效率和资源利用率。
  - `切分技巧 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/split_technique.html>`_：切分技巧是指通过手动配置某些关键算子的切分策略，减少张量重排布来提升训练效率。
  - `多副本 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/multiple_copy.html>`_：多副本是指在一个迭代步骤中，将一个训练batch拆分成多个，将模型并行通信与计算进行并发，提升资源利用率。

- 内存优化：内存优化包括重计算、数据集切分、Host&Device异构和异构存储，主要目标是节省内存空间。
  
  - `梯度累加 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/distributed_gradient_accumulation.html>`_：梯度累加通过在多个MicroBatch上计算梯度并将它们累加起来，然后一次性应用这个累积梯度来更新神经网络的参数。通过这种方法少量设备也能训练大Batch，有效减低内存峰值。
  - `重计算 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/recompute.html>`_：重计算是一种以时间换空间的技术，通过不保存某些正向算子的计算结果，以节省内存空间，在计算反向算子时，需要用到正向结果再重新计算正向算子。
  - `数据集切分 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/dataset_slice.html>`_：数据集单个数据过大甚至无法加载到单个设备的时候，可以对数据进行切分，进行分布式训练。数据集切分配合模型并行是有效降低显存占用的方式。
  - `Host&Device异构 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/host_device_training.html>`_：在遇到参数量超过Device内存上限的时候，可以把一些内存占用量大且计算量少的算子放在Host端，这样能同时利用Host端内存大，Device端计算快的特性，提升了设备的利用率。
  - `异构存储 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/memory_offload.html>`_：异构存储可以将暂时不需要用到的参数或中间结果拷贝到Host端内存或者硬盘，在需要时再恢复至Device端，从而减少显存占用。

- 通信优化：通信优化包括通信融合和通信子图提取与复用，主要目标是减少通信延时，提升性能。

  - `通信融合 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/comm_fusion.html>`_：通信融合可以将相同源节点和目标节点的通信算子合并到一次通信过程，避免多次通信带来额外开销。
  - `通信子图提取与复用 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/comm_subgraph.html>`_：通过对通信算子提取通信子图，替换原本的通信算子，可以减少通信耗时，同时减少模型编译时间。
