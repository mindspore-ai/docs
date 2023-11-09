# Distributed Parallelism Overview

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.3/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.3/tutorials/experts/source_en/parallel/overview.md)

In deep learning, as the size of the dataset and number of parameters grows, the time and hardware resources required for training will increase and eventually become a bottleneck that constrains training. Distributed parallel training, which can reduce the demand on hardware such as memory and computational performance, is an important optimization means to perform training. In addition, distributed parallel is important for large model training and inference, which provides powerful computational capabilities and performance advantages for handling large-scale data and complex models.

To implement distributed parallel training and inference, you can refer to the following guidelines:

## Startup Methods

MindSpore currently supports three startup methods:

- **Dynamic Networking**: Start via MindSpore internal dynamic grouping module, no dependency on external configurations or modules, Ascend/GPU/CPU support.
- **mpirun**: Start via the multi-process communication library OpenMPI, with Ascend/GPU support.
- **rank table**: After configuring the rank_table table, Ascend is supported by start scripts and processes corresponding to the number of cards.

Refer to the [Distributed Parallel Startup Methods](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/startup_method.html) section for details.

## Parallel Modes

Currently MindSpore can take the following parallel mode, and you can choose according to your needs:

- **Data Parallel Mode**: In data parallel mode, the dataset can be split in sample dimensions and distributed to different cards. If your dataset is large and the model parameters scale is able to operate on a single card, you can choose this parallel model. Refer to the [Data Parallel](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/data_parallel.html) tutorial for more information.
- **Automatic Parallel Mode**: a distributed parallel mode that combines data parallel and operator-level model parallel. It can automatically build cost models, find the parallel strategy with shorter training time, and select the appropriate parallel mode for the user. If your dataset and model parameters are large in size, and you want to configure the parallel strategy automatically, you can choose this parallel model. Refer to the [Automatic Parallelism](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/auto_parallel.html) tutorial for more information.
- **Semi-Automatic Parallel Mode**: Compared with automatic parallel, this mode requires the user to manually configure a slice strategy for the operators to realize parallel. If your dataset and model parameters are large, and you are familiar with the structure of the model, and know which "key operators" are prone to become computational bottlenecks to configure the appropriate slice strategy for the "key operators" to achieve better performance, you can choose this mode. Parallel mode. This mode also allows you to manually configure optimizer parallel and pipeline parallel. Refer to the [Semi-Automatic Parallel](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/semi_auto_parallel.html) tutorial for more information.
- **Manual Parallel Mode**: In manual parallel mode, you can manually implement parallel communication of models in distributed systems by transferring data based on communication operators such as `AllReduce`, `AllGather`, `Broadcast` and other communication operators. You can refer to the [Manual Parallel](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/manual_parallel.html) tutorial for more information.
- **Parameter Server Mode**: parameter servers offer better flexibility and scalability than synchronized training methods. You can refer to the [Parameter Server](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/parameter_server_training.html) mode tutorial for more information.

## Saving and Loading Models

Model saving can be categorized into merged and non-merged saving, which can be selected via the `integrated_save` parameter in `mindspore.save_checkpoint` or `mindspore.train.CheckpointConfig`. Model parameters are automatically aggregated and saved to the model file in merged save mode, while each card saves slices of the parameters on their respective cards in non-merged saving mode. You can refer to the [Model Saving](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/model_saving.html) tutorial for more information about model saving in each parallel mode.

Model loading can be categorized into full loading and slice loading. If the model file is saved with complete parameters, the model file can be loaded directly through the `load_checkpoint` interface. If the model file is a parameter-sliced file under multi-card, we need to consider whether the distributed slice strategy or cluster size has changed after loading. If the distributed slice strategy or cluster size remains unchanged, the corresponding parameter slice file for each card can be loaded via the `load_distributed_checkpoint` interface, which can be found in [model loading](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/model_loading.html) tutorial.

In the case that the saved and loaded distributed slice strategy or cluster size changes, the Checkpoint file under distribution needs to be converted to adapt to the new distributed slice strategy or cluster size. You can refer to [Model Transformation](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/model_transformation.html) for more information.

## Fault Recovery

During the distributed parallel training process, problems such as failures of computing nodes or communication interruptions may be encountered. MindSpore provides three recovery methods to ensure the stability and continuity of training:

- **Recovery based on full Checkpoint**：Before saving the Checkpoint file, the complete parameters of the model are aggregated by the AllGather operator, and the complete model parameter file is saved for each card, which can be loaded directly for recovery. Multiple checkpoints copies improve the fault tolerance of the model, but for large models, the aggregation process leads to excessive overhead of various resources. Refer to the [Model Loading](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/model_loading.html) tutorial for details.
- **Disaster Recovery in Dynamic Cluster Scenarios**: In dynamic cluster, if a process fails, the other processes will enter a waiting state, and the training task can be resumed by pulling up the fault process (only GPU hardware platforms are supported at present). Compared with other methods, this fault recovery method does not require restarting the cluster. For details, please refer to [Disaster Recovery in Dynamic Cluster Scenarios](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/disaster_recover.html) tutorial.
- **Recovery of redundant information based on parameter slicing**: In large model training, devices that are divided according to the dimension of data parallel have the same model parameters. According to this principle, these redundant parameter information can be utilized as a backup. When one node fails, another node utilizing the same parameters can recover the failed node. For details, please refer to the [Fault Recovery Based on Redundant Information](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/fault_recover.html) tutorial.

## Optimization Methods

If there is a requirement on performance, throughput, or scale, or if you don't know how to choose a parallel strategy, consider the following optimization techniques:

- **Parallel strategy optimization**：
    - **Strategy Selection**: Depending on the size of your model and the amount of data, you can refer to the [Strategy Selection](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/strategy_select.html) tutorial to select different parallel strategies to improve training efficiency and resource utilization.
    - **Slicing Techniques**: Slicing techniques are also key to achieving efficient parallel computation. In the [Slicing Techniques](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/split_technique.html) tutorial, you can learn how to apply a variety of slicing techniques to improve efficiency through concrete examples.
    - **Multi-copy Parallel**: Under the existing single-copy mode, certain underlying operators cannot perform computation at the same time when communicating, which leads to resource waste. Multi-copy mode slicing the data into multiple copies in accordance with the batch size dimension can make one copy communicate while the other copy performs computational operations, which improves the resource utilization rate. For details, please refer to the [Multi-copy Parallel](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/multiple_copy.html) tutorial.
- **Memory optimization**:
    - **Gradient Accumulation**: Gradient accumulation updates the parameters of a neural network by computing gradients over multiple MicroBatches and summing them up, then applying this accumulated gradient at once. In this way, a small number of devices can be trained on a large batch size, effectively minimizing memory spikes. For detailed information, refer to [Gradient Accumulation](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/distributed_gradient_accumulation.html) tutorial.
    - **Recompute**: Recompute saves memory space by not saving the result of the forward operators. When calculating the backward operators, you need to use the forward result before recalculating the forward operators. For details, please refer to the [recompute](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/recompute.html) tutorial.
    - **Dataset Slicing**: When a dataset is too large for a single piece of data, the data can be sliced for distributed training. Dataset slicing with model parallel is an effective way to reduce graphics memory usage. For details, please refer to the [dataset slicing](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/dataset_slice.html) tutorial.
    - **Host&Device Heterogeneous**: When the number of parameters exceeds the upper limit of Device memory, you can put some operators with large memory usage and small computation amount on the Host side, so that you can utilize the large memory on the Host side and the fast computation on the Device side at the same time, and improve the utilization rate of the device. For details, please refer to [Host&Device Heterogeneous](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/host_device_training.html) tutorial.
    - **Heterogeneous Storage**: large models are currently limited by the size of the graphics memory, making it difficult to train on a single card. In large-scale distributed cluster training, with communication becoming more and more costly, boosting the graphics memory of a single machine and reducing communication can also improve training performance. Heterogeneous storage can copy the parameters or intermediate results that are not needed temporarily to the memory or hard disk on the Host side, and then restore them to the Device side when needed. For details, please refer to [Heterogeneous Storage](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/memory_offload.html) tutorial.
- **Communication Optimization**：
    - **Communication fusion**: communication fusion can merge the communication operators of the same source and target nodes into a single communication process, avoiding the extra overhead of multiple communications. For details, please refer to [Communication Fusion](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/comm_fusion.html).
    - **Communication Subgraph Extraction and Reuse**: By extracting communication subgraphs for communication operators and replacing the original communication operators, we can reduce the communication time consumption and also reduce the model compilation time. For details, please refer to [Communication Subgraph Extraction and Reuse](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/comm_subgraph.html).

## Differences in Different Platforms

In distributed training, different hardware platforms (Ascend, CPU or GPU) support different characters, and users can choose the corresponding distributed startup method, parallel mode and optimization method according to their platforms.

### Differences in Startup Methods

- Ascend supports dynamic cluster, mpirun, and rank table startup.
- GPU supports dynamic cluster and mpirun startup.
- CPU only supports dynamic cluster startup.

For the detailed process, refer to [startup methods](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/startup_method.html).

### Differences in Parallel Methods

- Ascend and GPUs support all methods of parallel, including data parallel, semi-automatic parallel, automatic parallel, and more.
- CPU only supports data parallel.

For the detailed process, refer to [data parallel](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/data_parallel.html), [semi-automatic parallel](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/semi_auto_parallel.html), [auto-parallel](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/auto_parallel.html).

### Differences in Optimization Feature Support

- Ascend supports all optimization features.
- GPU support optimization features other than communication subgraph extraction and multiplexing.
- CPU does not support optimization features.

For the detailed process, refer to [optimization methods](https://www.mindspore.cn/tutorials/experts/en/r2.3/parallel/optimize_technique.html).

