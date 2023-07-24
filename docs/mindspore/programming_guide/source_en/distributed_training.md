# Distributed Training Overview

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/distributed_training.md)

In deep learning, the increasing number of datasets and parameters prolongs the training time and requires more hardware resources, becoming a training bottleneck. Parallel distributed training is an important optimization method for training, which can reduce requirements on hardware, such as memory and computing performance. Based on different parallel principles and modes, parallelism is generally classified into the following types:

- Data parallelism: splits data into many batches and then allocates the batches to each worker for model computation.
- Model parallelism: splits a model. MindSpore supports the intra-layer model parallelism. Parameters are split and then allocated to each worker for training.
- Hybrid parallelism: contains data parallelism and model parallelism.

MindSpore also provides the parallel distributed training function. It supports the following modes:

- `DATA_PARALLEL`: data parallelism.
- `SEMI_AUTO_PARALLEL`: semi-automatic parallelism. Users could set sharding strategies to operators to implement parallelism, which integrates data parallelism, model parallelism, and hybrid parallelism.
- `AUTO_PARALLEL`: automatic parallelism, which is under developing and only be validated in some specific networks. A cost model can be automatically created to find the parallel strategy with a relatively short training time and to select one parallel mode for users. MindSpore offers two different strategy search algorithms as follows:

    - `dynamic_programming`: Dynamic programming search algorithm. The optimal strategy of cost model description can be found, but it takes a long time to search for parallel strategy of huge network model. Its cost model refers to modeling the training time based on the memory-based computation and communication overheads of the Ascend 910 chip.
    - `recursive_programming`: Double recursive programming search algorithm. The optimal strategy can be generated instantly even for a large network or for a large-scale multi-device partitioning need. Its symbolic cost model can flexibly adapt to different accelerator clusters.

- `HYBRID_PARALLEL`: On MindSpore, users manually split parameters and base communication primitives to implement intra-layer model parallelism.
