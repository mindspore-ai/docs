Optimization Techniques
========================

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg
    :target: https://gitee.com/mindspore/docs/blob/br_base/tutorials/source_en/parallel/optimize_technique.rst
    :alt: View Source On Gitee

.. toctree::
  :maxdepth: 1
  :hidden:

  strategy_select
  split_technique
  multiple_copy
  high_dimension_tensor_parallel
  distributed_gradient_accumulation
  recompute
  dataset_slice
  host_device_training
  comm_fusion

Considering that in actual parallel training, there may be requirements for training performance, throughput or size, optimization can be considered in three ways: parallel strategy optimization, memory optimization and communication optimization

- Parallel Strategy Optimization: parallel strategy optimization mainly includes the selection of parallel strategy, sharding technique under operator-level parallel, and multi-copy technique.
  
  - `Strategy Selection <https://www.mindspore.cn/tutorials/en/br_base/parallel/strategy_select.html>`_: Depending on the model size and data volume size, different parallel strategies can be selected to improve training efficiency and resource utilization.
  - `Sharding Techniques <https://www.mindspore.cn/tutorials/en/br_base/parallel/split_technique.html>`_: The sharding technique refers to the reduction of tensor rearranging to improve training efficiency by manually configuring the sharding strategy for certain key operators.
  - `Multiply Copy <https://www.mindspore.cn/tutorials/en/br_base/parallel/multiple_copy.html>`_: Multi-copy refers to splitting a training batch into multiple ones in an iterative step to concurrently communicate and compute the model in parallel and improve resource utilization.
  - `High Dimension Tensor Parallel <https://www.mindspore.cn/tutorials/en/br_base/parallel/high_dimension_tensor_parallel.html>`_: High dimension tensor parallelism refers to multi-dimensional slicing of activation and weight tensor for MatMul computation in model parallelism, which reduces the communication amount and improves the training efficiency by optimizing the slicing strategy.

- Memory optimization: memory optimization includes gradient accumulation, recompute, dataset sharding, Host&Device heterogeneity and heterogeneous storage, with the main goal of saving memory space.
  
  - `Gradient Accumulation <https://www.mindspore.cn/tutorials/en/br_base/parallel/distributed_gradient_accumulation.html>`_: Gradient Accumulation updates the parameters of a neural network by computing gradients on multiple MicroBatches and summing them up, then applying this accumulated gradient at once. In this way a small number of devices can also train large Batches, effectively minimizing memory spikes.
  - `Recompute <https://www.mindspore.cn/tutorials/en/br_base/parallel/recompute.html>`_: Recomputation is a time-for-space technique that saves memory space by not saving the results of certain forward operator calculations, and when calculating the reverse operator, the forward results need to be used before recomputing the forward operator.
  - `Dataset Sharding <https://www.mindspore.cn/tutorials/en/br_base/parallel/dataset_slice.html>`_: When a dataset is too large individually or even cannot be loaded onto a single device, the data can be sliced for distributed training. Slicing the dataset with model parallel is an effective way to reduce the graphics memory usage.
  - `Host&Device Heterogeneous <https://www.mindspore.cn/tutorials/en/br_base/parallel/host_device_training.html>`_: When the number of parameters exceeds the upper limit of Device memory, you can put some operators with large memory usage and small computation on the Host side, which can simultaneously utilize the characteristics of large memory on the Host side and fast computation on the Device side, and improve the utilization rate of the device.

- Communication optimization: communication optimization includes communication fusion and communication subgraph extraction and multiplexing, and the main goal is to reduce communication delay and improve performance.

  - `Communication Fusion <https://www.mindspore.cn/tutorials/en/br_base/parallel/comm_fusion.html>`_: Communication fusion can merge the communication operators of the same source and target nodes into a single communication process, avoiding the extra overhead caused by multiple communications.
