# Distributed Parallelism Training

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/parallel_training.md)

## Parallel Modes and Application Scenarios

Large-scale deep learning model training requires robust computing power, especially in the case of a large dataset and a complex model architecture. As such, a single device usually cannot meet this requirement. To solve this problem, MindSpore provides a set of powerful parallelism strategies for configuration. You can use flexible parallelism strategies to greatly improve training efficiency and reduce computing resource consumption.

MindSpore offers parallel modes including data parallelism, model parallelism, pipeline parallelism, and sequence parallelism. They can be used independently or combined as a hybrid parallelism strategy to meet different model training requirements. By adopting proper parallelism strategies, you can leverage the computing resources of multiple devices, significantly improving the training efficiency.

In actual applications, different parallelism strategies apply to different scenarios.

- **Data parallelism**: applies to a simple model with a lot of data.
- **Model parallelism**: applies to a model with a huge number of parameters that a single device cannot accommodate.
- **Pipeline parallelism**: applies to ultra-large-scale model training that requires multi-device computing.
- **Sequence parallelism**: applies to a model with input of long sequences, reducing the GPU memory usage of a single device.
- **Multi-copy parallelism**: uses sequential scheduling algorithm to control the parallelism of fine-grained multi-branch operations, improving the overlap of computing and communications.
- **Optimizer parallelism**: distributes computing tasks of optimizers to multiple devices to reduce memory usage and improve training efficiency.

> The parallelism strategy configuration in the YAML file provided by the repository has been optimized. Currently, you are recommended to use semi-automatic parallelism for optimal performance and stability.

## Parallelism Features Supported by MindSpore Transformers

MindSpore Transformers supports multiple parallelism features. You can use these features to optimize the training of different model architectures and hardware configurations. The following table outlines these parallelism features and provides links to the details in the MindSpore documentation.

| **Parallelism Feature**                     | **Description**                                                                         |
|-----------------------------------|---------------------------------------------------------------------------------|
| **[Data parallelism](https://www.mindspore.cn/tutorials/en/master/parallel/data_parallel.html)**                    | Splits data to multiple devices and trains the data on each device at the same time. This mode applies to training a simple model with a lot of data.                                   |
| **[Model parallelism](https://www.mindspore.cn/tutorials/en/master/parallel/operator_parallel.html)**                    | Distributes model parameters to multiple devices. This mode applies to the scenario where a single device cannot accommodate the entire model.                                               |
| **[Pipeline parallelism](https://www.mindspore.cn/tutorials/en/master/parallel/pipeline_parallel.html)**                  | Divides an ultra-large model into multiple phases with each running on different devices for efficient training.                                       |
| **[Optimizer parallelism](https://www.mindspore.cn/tutorials/en/master/parallel/optimizer_parallel.html)**                  | Distributes the optimizer computation to multiple devices to reduce memory usage and improve training efficiency.                                                  |
| **Sequence parallelism**                     | Designed to share the memory and computation that cannot be sliced by model parallel, the inputs of LayerNorm and Dropout in the Transformer layer are sliced according to the sequence dimension to reduce the memory pressure on a single device.        |
| **[Long sequence parallelism](#long-sequence-parallelism)** | Slices all inputs and output activations by sequence to further reduce the GPU memory usage of the model for processing long sequence inputs.|
| **[Multi-copy parallelism](https://www.mindspore.cn/docs/en/master/features/parallel/pipeline_parallel.html#mindspore-interleaved-pipeline-scheduler)**                  | Implements fine-grained parallel control among multiple copies to optimize performance and resource utilization. This mode is suitable for efficient training of models with large specifications.                                    |

For details about how to configure distributed parallel parameters, see [MindSpore Transformers Configuration Description](https://www.mindspore.cn/mindformers/docs/en/dev/feature/configuration.html).

## Introduction to Parallel Characterization

### Long Sequence Parallelism

From generative AI to scientific models, long sequence training is becoming very important. Existing parallel methods such as data, tensor and pipelining cannot slice in the sequence dimension. As the sequence dimension (S) grows, the training memory overhead grows at the rate of O($S^2$). Sequence parallelism slices all inputs and output activations in the sequence dimension, which is used to reduce the limitations on the length of the input sequences and efficiently support ultra-long sequence training.

#### Ring Attention Sequence Parallelism

Long Sequence Parallel Algorithm, Ring Attention, is a representative technique for long sequence parallelism in the current industry, which is used to solve the memory overhead problem during long sequence training, while realizing computation and communication masking. The Ring Attention algorithm utilizes the chunking property of Attention, when the sequence parallelism is N, Q, K, V are sliced into N sub-chunks, and each card calls the Flash Attention algorithm to compute the Attention result of the local QKV sub-chunks respectively. Since each card only needs to compute the Attention of the sliced QKV sub-chunks, its memory occupation is reduced significantly. Ring Attention uses ring communication to collect and send sub-chunks to neighboring cards while doing FA computation to maximize the masking of computation and communication, which guarantees the overall performance of long sequence parallelism.

MindSpore Transformers has support for configuring Ring Attention sequence parallel schemes, which can be enabled with the following configuration item:

```yaml
model:
  model_config:
    ...
    use_ring_attention: True
    ...
parallel_config:
  ...
  context_parallel: 2
  ...
```

Parameter Descriptions:

- use_ring_attention: Whether to enable Ring Attention, default is False.
- context_parallel:  The number of sequence parallel slices, default is 1, configure according to user requirements.

For configuration method of distributed parallel parameters, refer to the contents of the Parallel Configuration section in [MindSpore Transformers configuration description](https://www.mindspore.cn/mindformers/docs/en/dev/feature/configuration.html).

#### Ulysses Sequence Parallelism

The [Ulysses long sequence parallelism scheme](https://arxiv.org/abs/2309.14509) proposed by DeepSpeed slices the individual samples in the seq dimension to different compute cards; then, prior to the attention computation, an all-to-all communication operation is performed on the QKVs to allow each compute card to receive the complete sequence, allowing each computation card to compute different attention heads in parallel. Finally, another all-to-all is used after the ATTENTION computation to collect results on the attention head while re-slicing on the seq dimension. This scheme effectively extends the length of the trained sequences while keeping the communication relatively low.

MindSpore Transformers has support for configuring the Ulysses Sequence Parallel Scheme, which can be enabled with the following configuration item:

```yaml
model:
  model_config:
    ...
    use_attn_mask_compression: True # Enable attention_mask compression
    ...
parallel:
  ...
  enable_alltoall: True  # Allow inputting of alltoall operator
  ...
parallel_config:
  ...
  context_parallel: 2
  context_parallel_algo: ulysses_cp  # Enable Ulysses sequence parallelism
  ...
```

Parameter Descriptions:

- use_attn_mask_compression: Whether to mask the Score matrix in Self-Attention, default is False, it is recommended to turn it on to reduce the video memory usage in Ulysses sequence parallel scheme.
- enable_alltoall: Generate alltoall communication operator, default is False, when the parameter is not enabled, it will be replaced by a combination of other operators such as allgather. See MindSpore `set_auto_parallel_context` [interface documentation](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_auto_parallel_context.html). We expect to be able to directly input allto_all communication operators when we enable the Ulysses scenario, so we turn this configuration item on.
- context_parallel_algo: Set to `ulysses_cp` to enable Ulysses sequence parallelism.

For configuration method of distributed parallel parameters, refer to the contents of the Parallel Configuration section in [MindSpore Transformers configuration description](https://www.mindspore.cn/mindformers/docs/en/dev/feature/configuration.html).

#### Hybrid Sequence Parallelism

Currently, both Ulysses and Ring Attention sequence parallel schemes have certain limitations. Although Ring Attention sequence parallel scheme can theoretically expand the sequence length infinitely, the communication and computation bandwidth utilization is low, and the performance is inferior to that of Ulysses sequence parallel scheme when the sequence block size is low. The sequence parallelism of Ulysses in GQA and MQA scenarios is limited by the number of Heads and the expansion of sequence length is limited. Hybrid sequence parallelism fuses Ulysses and Ring Attention sequence parallelism scheme, which can solve the above defects.

MindSpore Transformers has support for configuring hybrid sequence parallel schemes, which can be enabled with the following configuration items:

```yaml
parallel:
  ...
  enable_alltoall: True  # Allow inputting of alltoall operator
  ...
parallel_config:
  ...
  context_parallel: 16
  context_parallel_algo: hybrid_cp  # Enable hybrid sequence parallel
  ulysses_degree_in_cp: 8
  ...
```

Parameter Descriptions:

- context_parallel_algo: hybrid sequence parallelism is turned on when set to `hybrid_cp`.
- ulysses_degree_in_cp: the number of parallel slices of the Ulysses sequence.

For configuration method of distributed parallel parameters, refer to the contents of the Parallel Configuration section in [MindSpore Transformers configuration description](https://www.mindspore.cn/mindformers/docs/en/dev/feature/configuration.html).

### Pipeline Parallelism

#### Sequence Pipeline Parallelism (Seq-Pipe)

The model inputs are segmented along the sequence dimension and unfolded into multiple sequence chunks. In the original 1F1B (One Forward One Backward) and 1F1B-Interleave methods, the scheduling unit is reduced to a Sequence Chunk. `seq_split_num` represents the number of Sequence Chunk; when `seq_split_num`=1, it degrades to 1F1B or 1F1B-Interleave.

MindSpore Transformers supports configuring the Seq-Pipe pipeline parallelism, which can be enabled through the following configuration items:

```yaml
# parallel context
parallel:
  pipeline_config:
    pipeline_interleave: true
    pipeline_scheduler: 'seqpipe'

# parallel config
parallel_config:
  seq_split_num: 2
```

Parameter Descriptions:

- pipeline_scheduler: The scheduling strategy for the pipeline, currently, mindspore transformers only supports setting this to `"seqpipe"`.
- seq_split_num: The number of Sequence Chunk which splits along the sequence dimension of the input.

Notes:

- Currently, only Llama and DeepSeek series models are supported.
- Using Megatron's multi-source datasets for training is not yet supported.

For more information on configuring distributed parallel parameters, see the [MindSpore Transformers configuration description](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/configuration.html), specifically the section on parallel configuration.

## MindSpore Transformers Distributed Parallel Application Practices

In the [Llama3_1-70B fine-tuning configuration](https://gitee.com/mindspore/mindformers/blob/dev/research/llama3_1/llama3_1_70b/finetune_llama3_1_70b.yaml#) file provided on the official website, multiple distributed parallelism strategies are used to improve the training efficiency in the multi-node multi-device environment. The main parallelism strategies and key parameters involved in the configuration file are as follows:

- **Data parallelism**: No additional data parallelism is enabled (`data_parallel: 1`).
- **Model parallelism**: A model is sliced into eight parts, which are computed on different devices (`model_parallel: 8`).
- **Pipeline parallelism**: A model is divided into eight pipeline phases, which run on different devices in sequence (`pipeline_stage: 8`).
- **Sequence parallelism**: After it is enabled (`use_seq_parallel: True`), the inputs of LayerNorm and Dropout at the Transformer layer are sliced by sequence. In this way, each device only needs to process part of LayerNorm and Dropout, reducing the model GPU memory usage.
- **Multi-copy parallelism**: Sequential scheduling algorithm is used to control the parallelism of fine-grained multi-branch operations (`fine_grain_interleave: 2`), improving the overlap of computing and communications.
- **Optimizer parallelism**: The calculation of optimizers is distributed to multiple devices to reduce memory usage (`enable_parallel_optimizer: True`).

> Sequential parallelism must be turned on at the same time that fine-grained multicopy parallelism is turned on.

With the preceding configurations, the distributed training on Llama3_1-70B can effectively utilize hardware resources in a multi-node multi-device environment to implement efficient and stable model training.
