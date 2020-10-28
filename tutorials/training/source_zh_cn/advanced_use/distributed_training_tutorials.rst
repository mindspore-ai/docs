分布式并行训练
===============

在深度学习中，当数据集和参数量的规模越来越大，训练所需的时间和硬件资源会随之增加，最后会变成制约训练的瓶颈。分布式并行训练，可以降低对内存、计算性能等硬件的需求，是进行训练的重要优化手段。根据并行的原理及模式不同，业界主流的并行类型有以下几种：

- 数据并行（Data Parallel）：对数据进行切分的并行模式，一般按照batch维度切分，将数据分配到各个计算单元（worker）中，进行模型计算。
- 模型并行（Model Parallel）：对模型进行切分的并行模式。MindSpore中支持层内模型并行模式，即对参数切分后分配到各个计算单元中进行训练。
- 混合并行（Hybrid Parallel）：指涵盖数据并行和模型并行的并行模式。

当前MindSpore也提供分布式并行训练的功能。它支持了多种模式包括：

- `DATA_PARALLEL`：数据并行模式。
- `AUTO_PARALLEL`：自动并行模式，融合了数据并行、模型并行及混合并行的1种分布式并行模式，可以自动建立代价模型，为用户选择1种并行模式。其中，代价模型指围绕Ascend 910芯片基于内存的计算开销和通信开销对训练时间建模，并设计高效的算法找到训练时间较短的并行策略。
- `SEMI_AUTO_PARALLEL`：半自动并行模式，相较于自动并行，该模式需要用户对算子手动配置切分策略实现并行。
- `HYBRID_PARALLEL`：在MindSpore中特指用户通过手动切分模型实现混合并行的场景。

.. toctree::
  :maxdepth: 1

  distributed_training_ascend
  distributed_training_gpu
  apply_host_device_training
  apply_parameter_server_training
  save_load_model_hybrid_parallel
