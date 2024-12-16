# 分布式并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindformers/docs/source_zh_cn/function/distributed_parallel.md)

## 并行模式与应用场景

在大规模深度学习模型的训练中，尤其是面对庞大的数据集和复杂的模型架构时，单一设备的算力往往不足以应对这种需求。为了解决这个问题，MindSpore 提供了一套强大的并行策略配置，通过灵活的并行策略可以大幅提升训练效率，并降低计算资源的消耗。

MindSpore 的并行模式包括数据并行、模型并行、流水线并行、序列并行等，这些模式可以单独使用，也可以结合在一起，形成复杂的混合并行策略，以应对不同的模型训练需求。通过合理配置这些并行策略，开发者可以有效利用多设备的计算资源，极大地提升训练效率。

在实际应用中，不同的并行策略适用于不同的场景：

- **数据并行**：适用于数据量大，模型相对简单的场景。
- **模型并行**：适用于模型参数量巨大，单个设备无法容纳整个模型的场景。
- **流水线并行**：适用于超大规模模型训练，需多设备共同计算的场景。
- **序列并行**：适用于长序列输入的模型，减少单设备显存占用的场景。
- **多副本并行**：通过执行序调度算法控制细粒度多分支的并行，提高计算与通信的相互掩盖。
- **优化器并行**：将优化器的计算任务分散到多个设备上，以减少内存占用并提高训练效率。

> 仓库中提供的 YAML 文件中并行策略配置已经优化，当前推荐用户使用半自动并行，以确保最佳性能和稳定性。

## MindFormers 支持的并行特性

MindFormers 支持多种并行特性，开发者可以利用这些特性来优化不同模型架构和硬件配置的训练。以下表格概述了这些并行特性，并提供了指向 MindSpore 文档中详细说明的链接。

| **并行特性**                      | **描述**                                                                          |
|-----------------------------------|---------------------------------------------------------------------------------|
| **[数据并行](https://www.mindspore.cn/docs/zh-CN/r2.4.10/model_train/parallel/data_parallel.html)**                     | 将数据拆分到多个设备上，并在每个设备上同时进行训练。适用于数据量大且模型相对简单的任务。                                    |
| **[模型并行](https://www.mindspore.cn/docs/zh-CN/r2.4.10/model_train/parallel/operator_parallel.html)**                     | 将模型参数分布到多个设备上，适合单个设备无法容纳整个模型的情况。                                                |
| **[流水线并行](https://www.mindspore.cn/docs/zh-CN/r2.4.10/model_train/parallel/pipeline_parallel.html)**                   | 将模型分割成多个阶段，每个阶段在不同的设备上运行，以实现超大规模模型的高效训练。                                        |
| **[优化器并行](https://www.mindspore.cn/docs/zh-CN/r2.4.10/model_train/parallel/optimizer_parallel.html)**                   | 将优化器计算分布到多个设备上，减少内存占用，提高训练效率。                                                   |
| **[序列并行](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/feature_cards/Long_Sequence_Training.md)**                     | 设计用于处理长序列输入的模型，将Transformer层中的LayerNorm及Dropout的输入按照序列维度进行切分，减少单设备的显存压力。        |
| **上下文并行**  | 设计用于处理长序列输入的模型，对所有的input输入和所有的输出activation在sequence维度上进行切分，对于超长序列输入场景进一步减少显存占用。 |
| **[多副本并行](https://www.mindspore.cn/docs/zh-CN/r2.4.10/model_train/parallel/pipeline_parallel.html#mindspore%E4%B8%AD%E7%9A%84interleaved-pipeline%E8%B0%83%E5%BA%A6)**                   | 用于在多个副本之间实现精细的并行控制，优化性能和资源利用率，适合大规格模型的高效训练。                                     |

关于分布式并行参数的配置方法，参见 [MindFormers 配置说明](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/appendix/conf_files.html) 中的并行配置章节下的具体内容。

## MindFormers 分布式并行应用实践

在官网提供的[Llama3-70B微调配置](https://gitee.com/kong_de_shu/mindformers/blob/dev/research/llama3/finetune_llama3_70b.yaml#)文件中，使用了多种分布式并行策略，以提升多机多卡环境中的训练效率。以下是该配置文件中涉及的主要并行策略和关键参数：

- **数据并行**：未启用额外的数据并行（`data_parallel: 1`）。
- **模型并行**：模型被切分成8个部分，在不同设备上计算（`model_parallel: 8`）。
- **流水线并行**：模型分为8个流水线阶段，按顺序在不同设备上运行（`pipeline_stage: 8`）。
- **序列并行**：开启序列并行（`use_seq_parallel: True`），将Transformer层中的LayerNorm及Dropout的输入按照序列维度进行切分，使各设备只需处理部分的LayerNorm和Dropout，减少模型显存占用。
- **多副本并行**：通过执行序调度算法控制细粒度多分支的并行（`fine_grain_interleave: 2`），提高计算与通信的相互掩盖。
- **优化器并行**：优化器计算分散到多个设备上，以减少内存占用（`enable_parallel_optimizer: True`）。

> 注意：开启细粒度多副本并行的同时必须开启序列并行。

通过以上配置，Llama3-70B的分布式训练在多机多卡环境中可以有效利用硬件资源，实现高效、稳定的模型训练。
