分布式并行总览
================

在深度学习中，当数据集和参数量的规模越来越大，训练所需的时间和硬件资源会随之增加，最后会变成制约训练的瓶颈。分布式并行训练，可以降低对内存、计算性能等硬件的需求，是进行训练的重要优化手段。根据并行的原理及模式不同，业界主流的并行类型有以下几种：

- 数据并行（Data Parallel）：对数据进行切分的并行模式，一般按照batch维度切分，将数据分配到各个计算单元（worker）中，进行模型计算。
- 模型并行（Model Parallel）：对模型进行切分的并行模式。模型并行可分为：算子级模型并行、流水线模型并行、优化器模型并行等。
- 混合并行（Hybrid Parallel）：指涵盖数据并行和模型并行的并行模式。

当前MindSpore也提供分布式并行训练的功能。它支持了多种模式包括：

- `DATA_PARALLEL`：数据并行模式。
- `AUTO_PARALLEL`：自动并行模式，融合了数据并行、算子级模型并行的分布式并行模式，可以自动建立代价模型，找到训练时间较短的并行策略，为用户选择合适的并行模式。当前MindSpore支持算子级并行策略的自动搜索，提供了如下的三种不同的策略搜索算法：

  - `dynamic_programming`：动态规划策略搜索算法。能够搜索出代价模型刻画的最优策略，但在搜索巨大网络模型的并行策略时耗时较长。其代价模型是围绕Ascend 910芯片基于内存的计算开销和通信开销对训练时间建模。
  - `recursive_programming`：双递归策略搜索算法。对于巨大网络以及大规模多卡切分能够保证瞬间生成最优策略。其基于符号运算的代价模型可以自由适配不同的加速器集群。
  - `sharding_propagation`：切分策略传播算法。由配置并行策略的算子向未配置的算子传播并行策略。在传播时，算法会尽量选取引发张量重排布通信最少的策略。关于算子的并行策略配置和张量重排布，可参考这篇[设计文档](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/design/distributed_training_design.html#自动并行原理)。

- `SEMI_AUTO_PARALLEL`：半自动并行模式，相较于自动并行，该模式需要用户对算子手动配置切分策略实现并行。
- `HYBRID_PARALLEL`：在MindSpore中特指用户通过手动切分模型实现混合并行的场景。

.. toctree::
  :maxdepth: 1

  auto_parallel
  distributed_training_mode
  distributed_training_ops