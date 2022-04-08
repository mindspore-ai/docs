Distributed Parallel Overview
==============================

In deep learning, the increasing number of dataset and parameters prolongs the training time and requires more hardware resources, which becomes a training bottleneck. Parallel training is an important optimization method, which reduces requirements of a single device for high memory and performance. Parallelisms are generally classified into the following types:

- Data parallelism: splits data into many batches and then allocates the batches to each device for model computation.
- Model parallelism: splits the model across multiple devices, which includes op-level model parallelism, pipeline model parallelism and optimizer model parallelism.
- Hybrid parallelism: contains data parallelism and model parallelism.

MindSpore also provides the parallel training functionality. It supports the following modes:

- `DATA_PARALLEL`: data parallelism.
- `AUTO_PARALLEL`: automatic parallelism, which integrates data parallelism, model parallelism, and hybrid parallelism. A cost model is built to characterize training time and memory usage. Currently, MindSpore supports searching strategies for op-level model parallelism, which includes three different algorithms as follows:

  - `dynamic_programming`: Dynamic programming search algorithm. The optimal strategy under the cost model description can be found, but it takes a long time to search for parallel strategy of huge network model. Its cost model refers to modeling the training time based on the memory-based computation and communication overheads of the Ascend 910 chip.
  - `recursive_programming`: Double recursive programming search algorithm. The optimal strategy can be generated instantly even for a large network. Its symbolic cost model can be flexibly adapted to different accelerator clusters.
  - `sharding_propagation`: Sharding Propagation algorithms. This mode requires users to configure sharding strategies for some operators, and then propagates from these operators to other operators, with the goal of minimizing communication cost in tensor redistribution. For definitions of sharding strategy and tensor redistribution, please refer to this [design article](https://www.mindspore.cn/docs/en/master/design/distributed_training_design.html#id10)

- `HYBRID_PARALLEL`: On MindSpore, users manually split parameters to implement intra-layer model parallelism.

.. toctree::
  :maxdepth: 1

  auto_parallel
  distributed_training_mode