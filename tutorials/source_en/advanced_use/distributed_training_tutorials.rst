Distributed training
====================

  In deep learning, the increasing number of datasets and parameters prolongs the training time and requires more hardware resources, becoming a training bottleneck. Parallel distributed training is an important optimization method for training, which can reduce requirements on hardware, such as memory and computing performance. Based on different parallel principles and modes, parallelism is generally classified into the following types:

- Data parallelism: splits data into many batches and then allocates the batches to each worker for model computation.
- Model parallelism: splits a model. MindSpore supports the intra-layer model parallelism. Parameters are split and then allocated to each worker for training.
- Hybrid parallelism: contains data parallelism and model parallelism.

MindSpore also provides the parallel distributed training function. It supports the following modes:

- `DATA_PARALLEL`: data parallelism.
- `AUTO_PARALLEL`: automatic parallelism, which integrates data parallelism, model parallelism, and hybrid parallelism. A cost model can be automatically created to select one parallel mode for users. Creating a cost model refers to modeling the training time based on the memory-based computation and communication overheads of the Ascend 910 chip, and designing efficient algorithms to develop a parallel strategy with a relatively short training time.
- `HYBRID_PARALLEL`: On MindSpore, users manually split parameters to implement intra-layer model parallelism.

.. toctree::
  :maxdepth: 1

  distributed_training_ascend
  host_device_training
  checkpoint_for_hybrid_parallel
