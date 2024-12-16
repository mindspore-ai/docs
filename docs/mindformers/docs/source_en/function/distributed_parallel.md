# Distributed Parallelism

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindformers/docs/source_en/function/distributed_parallel.md)

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

## Parallelism Features Supported by MindFormers

MindFormers supports multiple parallelism features. You can use these features to optimize the training of different model architectures and hardware configurations. The following table outlines these parallelism features and provides links to the details in the MindSpore documentation.

| **Parallelism Feature**                     | **Description**                                                                         |
|-----------------------------------|---------------------------------------------------------------------------------|
| **[Data parallelism](https://www.mindspore.cn/docs/en/r2.4.10/model_train/parallel/data_parallel.html)**                    | Splits data to multiple devices and trains the data on each device at the same time. This mode applies to training a simple model with a lot of data.                                   |
| **[Model parallelism](https://www.mindspore.cn/docs/en/r2.4.10/model_train/parallel/operator_parallel.html)**                    | Distributes model parameters to multiple devices. This mode applies to the scenario where a single device cannot accommodate the entire model.                                               |
| **[Pipeline parallelism](https://www.mindspore.cn/docs/en/r2.4.10/model_train/parallel/pipeline_parallel.html)**                  | Divides an ultra-large model into multiple phases with each running on different devices for efficient training.                                       |
| **[Optimizer parallelism](https://www.mindspore.cn/docs/en/r2.4.10/model_train/parallel/optimizer_parallel.html)**                  | Distributes the optimizer computation to multiple devices to reduce memory usage and improve training efficiency.                                                  |
| **[Sequence parallelism](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/feature_cards/Long_Sequence_Training.md)**                    | Slices the LayerNorm and Dropout inputs at the Transformer layer by sequence to reduce the GPU memory pressure of a single device. This mode applies to a model for processing long sequence inputs.       |
| **Context parallelism** | Slices all inputs and output activations by sequence to further reduce the GPU memory usage of the model for processing long sequence inputs.|
| **[Multi-copy parallelism](https://www.mindspore.cn/docs/en/r2.4.10/model_train/parallel/pipeline_parallel.html#mindspore-interleaved-pipeline-scheduler)**                  | Implements fine-grained parallel control among multiple copies to optimize performance and resource utilization. This mode is suitable for efficient training of models with large specifications.                                    |

For details about how to configure distributed parallel parameters, see [MindFormers Configuration Description](https://www.mindspore.cn/mindformers/docs/en/r1.3.2/appendix/conf_files.html).

## MindFormers Distributed Parallel Application Practices

In the [Llama3-70B fine-tuning configuration](https://gitee.com/kong_de_shu/mindformers/blob/dev/research/llama3/finetune_llama3_70b.yaml#) file provided on the official website, multiple distributed parallelism strategies are used to improve the training efficiency in the multi-node multi-device environment. The main parallelism strategies and key parameters involved in the configuration file are as follows:

- **Data parallelism**: No additional data parallelism is enabled (`data_parallel: 1`).
- **Model parallelism**: A model is sliced into eight parts, which are computed on different devices (`model_parallel: 8`).
- **Pipeline parallelism**: A model is divided into eight pipeline phases, which run on different devices in sequence (`pipeline_stage: 8`).
- **Sequence parallelism**: After it is enabled (`use_seq_parallel: True`), the inputs of LayerNorm and Dropout at the Transformer layer are sliced by sequence. In this way, each device only needs to process part of LayerNorm and Dropout, reducing the model GPU memory usage.
- **Multi-copy parallelism**: Sequential scheduling algorithm is used to control the parallelism of fine-grained multi-branch operations (`fine_grain_interleave: 2`), improving the overlap of computing and communications.
- **Optimizer parallelism**: The calculation of optimizers is distributed to multiple devices to reduce memory usage (`enable_parallel_optimizer: True`).

> Note: Sequential parallelism must be turned on at the same time that fine-grained multicopy parallelism is turned on.

With the preceding configurations, the distributed training on Llama3-70B can effectively utilize hardware resources in a multi-node multi-device environment to implement efficient and stable model training.
