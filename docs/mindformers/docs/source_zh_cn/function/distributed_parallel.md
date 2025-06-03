# 分布式并行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/function/distributed_parallel.md)

## 并行模式与应用场景

在大规模深度学习模型的训练中，尤其是面对庞大的数据集和复杂的模型架构时，单一设备的算力往往不足以应对这种需求。为了解决这个问题，MindSpore 提供了一套强大的并行策略配置，通过灵活的并行策略可以大幅提升训练效率，并降低计算资源的消耗。

MindSpore 的并行模式包括数据并行、模型并行、流水线并行、序列并行等。这些模式可以单独使用，也可以结合在一起，形成复杂的混合并行策略，以应对不同的模型训练需求。通过合理配置这些并行策略，开发者可以有效利用多设备的计算资源，极大地提升训练效率。

在实际应用中，不同的并行策略适用于不同的场景：

- **数据并行**：适用于数据量大，模型相对简单的场景。
- **模型并行**：适用于模型参数量巨大，单个设备无法容纳整个模型的场景。
- **流水线并行**：适用于超大规模模型训练，需多设备共同计算的场景。
- **序列并行**：适用于长序列输入的模型，减少单设备显存占用的场景。
- **多副本并行**：通过执行序调度算法控制细粒度多分支的并行，提高计算与通信的相互掩盖。
- **优化器并行**：将优化器的计算任务分散到多个设备上，以减少内存占用并提高训练效率。

> 仓库中提供的 YAML 文件中并行策略配置已经优化，当前推荐用户使用半自动并行，以确保最佳性能和稳定性。

## MindSpore Transformers 支持的并行特性

MindSpore Transformers 支持多种并行特性，开发者可以利用这些特性来优化不同模型架构和硬件配置的训练。以下表格概述了这些并行特性，并提供了指向 MindSpore 文档中详细说明的链接。

| **并行特性**                      | **描述**                                                                          |
|-----------------------------------|---------------------------------------------------------------------------------|
| **[数据并行](https://www.mindspore.cn/docs/zh-CN/master/features/parallel/data_parallel.html)**                     | 将数据拆分到多个设备上，并在每个设备上同时进行训练。适用于数据量大且模型相对简单的任务。                                    |
| **[模型并行](https://www.mindspore.cn/docs/zh-CN/master/features/parallel/operator_parallel.html)**                     | 将模型参数分布到多个设备上，适合单个设备无法容纳整个模型的情况。                                                |
| **[流水线并行](https://www.mindspore.cn/docs/zh-CN/master/features/parallel/pipeline_parallel.html)**                   | 将模型分割成多个阶段，每个阶段在不同的设备上运行，以实现超大规模模型的高效训练。                                        |
| **[优化器并行](https://www.mindspore.cn/docs/zh-CN/master/features/parallel/optimizer_parallel.html)**                   | 将优化器计算分布到多个设备上，减少内存占用，提高训练效率。                                                   |
| **序列并行**                     | 设计用于分摊模型并行无法切分的显存和计算，将Transformer层中的LayerNorm及Dropout的输入按照序列维度进行切分，减少单设备的显存压力。        |
| **[长序列并行](#长序列并行)**  | 设计用于处理长序列输入的模型，对所有的input输入和所有的输出activation在sequence维度上进行切分，对于超长序列输入场景进一步减少显存占用。 |
| **[多副本并行](https://www.mindspore.cn/docs/zh-CN/master/features/parallel/pipeline_parallel.html#mindspore%E4%B8%AD%E7%9A%84interleaved-pipeline%E8%B0%83%E5%BA%A6)**                   | 用于在多个副本之间实现精细的并行控制，优化性能和资源利用率，适合大规格模型的高效训练。                                     |

关于分布式并行参数的配置方法，参见 [MindSpore Transformers 配置说明](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/appendix/conf_files.html) 中的并行配置章节下的具体内容。

## 并行特性介绍

### 长序列并行

从生成性AI到科研模型，长序列训练正在变得非常重要。现有的数据、张量和流水线等并行方法无法在序列维度进行切分。当序列维度（S）增长时，训练内存开销会以O（$S^2$）的速度增长。序列并行对所有的input输入和所有的输出activation在sequence维度上进行切分，用于减少输入序列长度的限制，有效地支持超长序列训练。

#### Ring Attention序列并行

长序列并行算法 Ring Attention 是当前业界长序列并行的代表性技术，用于解决长序列训练时的内存开销问题，同时实现计算与通信掩盖。Ring Attention 算法利用 Attention 的分块计算性质，当序列并行度为 N 时，将 Q，K，V 分别切分为 N 个子块，每张卡分别调用 Flash Attention 算子来计算本地 QKV 子块的 Attention 结果。由于每张卡只需要计算切分后 QKV 子块的 Attention，其内存占用大幅降低。Ring Attention 在做 FA 计算的同时采用环形通信向相邻卡收集和发送子块，实现计算与通信的最大化掩盖，保障了长序列并行的整体性能。

MindSpore Transformers已支持配置Ring Attention序列并行方案，可通过以下配置项使能：

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

参数说明：

- use_ring_attention：是否开启Ring Attention，默认为False。
- context_parallel：序列并行切分数量，默认为1，根据用户需求配置。

关于分布式并行参数的配置方法，参见 [MindSpore Transformers 配置说明](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/appendix/conf_files.html) 中的并行配置章节下的具体内容。

#### Ulysses序列并行

DeepSpeed提出的[Ulysses长序列并行方案](https://arxiv.org/abs/2309.14509)，将各个样本在seq维度切分给不同的计算卡；然后，在attention计算之前，对QKV执行all-to-all通信操作，以使每个计算卡接收完整的序列，使得各计算卡可以并行计算不同的注意力头；最后，在attention计算后使用另一个all-to-all来在注意力头上收集结果，同时重新在seq维度上进行切分。该方案可以有效扩展训练的序列长度，同时保持相对较低的通信量。

MindSpore Transformers已支持配置Ulysses序列并行方案，可通过以下配置项使能：

```yaml
model:
  model_config:
    ...
    use_attn_mask_compression: True #使能attention_mask压缩
    ...
parallel:
  ...
  enable_alltoall: True  # 允许插入alltoall算子
  ...
parallel_config:
  ...
  context_parallel: 2
  context_parallel_algo: ulysses_cp  # 使能Ulysses序列并行
  ...
```

参数说明：

- use_attn_mask_compression：是否对Self-Attention中的Score矩阵进行掩码操作，默认为False，Ulysses序列并行方案下建议开启减少显存占用。
- enable_alltoall：生成alltoall通信算子，默认为False，不启用时将会由allgather等其他算子组合完成等价替代，可参考MindSpore `set_auto_parallel_context`[接口文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_auto_parallel_context.html)；启用Ulysses方案时我们期望能够直接插入alltoall通信算子，因此将该配置项打开。
- context_parallel_algo：设置为`ulysses_cp`开启Ulysses序列并行。

关于分布式并行参数的配置方法，参见 [MindSpore Transformers 配置说明](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/appendix/conf_files.html) 中的并行配置章节下的具体内容。

#### 混合序列并行

目前Ulysses和Ring Attention序列并行方案均存在一定局限性，Ring Attention序列并行方案虽然理论上序列长度能够无限拓展，但通信和计算带宽利用率较低，在序列块大小较低时性能劣于Ulysses序列并行方案。而Ulysses在GQA、MQA场景下的序列并行受Head数量限制，序列长度的扩展有限。混合序列并行融合了Ulysses和Ring Attention序列并行方案，可以解决上述缺陷。

MindSpore Transformers已支持配置混合序列并行方案，可通过以下配置项使能：

```yaml
parallel:
  ...
  enable_alltoall: True  # 允许插入alltoall算子
  ...
parallel_config:
  ...
  context_parallel: 16
  context_parallel_algo: hybrid_cp  # 使能混合序列并行
  ulysses_degree_in_cp: 8
  ...
```

参数说明：

- context_parallel_algo：设置为`hybrid_cp`时开启混合序列并行。
- ulysses_degree_in_cp：Ulysses序列并行切分数量。

关于分布式并行参数的配置方法，参见 [MindSpore Transformers 配置说明](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/appendix/conf_files.html) 中的并行配置章节下的具体内容。

### 流水线并行

#### 序列流水线并行（Seq-Pipe）

模型输入按sequence维度进行切分，展开为多个序列块（Sequence Chunk）。在原有的1F1B和1F1B-Interleave上，将调度单位缩小为Sequence Chunk。`seq_split_num`为切分个数，当`seq_split_num`=1时，退化为1F1B或1F1B-Interleave。

MindSpore Transformers已支持配置Seq-Pipe流水线并行方案，可通过以下配置项使能：

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

参数说明：

- pipeline_scheduler：流水线的调度策略，目前mindformers只支持设置为`"seqpipe"`。
- seq_split_num：输入按序列维度的切分个数。

注意：

- 目前仅支持Llama和DeepSeek系列模型。
- 目前暂不支持使用Megatron的多源数据集进行训练的场景。

关于分布式并行参数的配置方法，参见 [MindSpore Transformers配置说明](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/appendix/conf_files.html) 中的并行配置章节下的具体内容。

## MindSpore Transformers 分布式并行应用实践

在官网提供的[Llama3-70B微调配置](https://gitee.com/kong_de_shu/mindformers/blob/dev/research/llama3/llama3_70b/finetune_llama3_70b.yaml#)文件中，使用了多种分布式并行策略，以提升多机多卡环境中的训练效率。以下是该配置文件中涉及的主要并行策略和关键参数：

- **数据并行**：未启用额外的数据并行（`data_parallel: 1`）。
- **模型并行**：模型被切分成8个部分，在不同设备上计算（`model_parallel: 8`）。
- **流水线并行**：模型分为8个流水线阶段，按顺序在不同设备上运行（`pipeline_stage: 8`）。
- **序列并行**：开启序列并行（`use_seq_parallel: True`），将Transformer层中的LayerNorm及Dropout的输入按照序列维度进行切分，使各设备只需处理部分的LayerNorm和Dropout，减少模型显存占用。
- **多副本并行**：通过执行序调度算法控制细粒度多分支的并行（`fine_grain_interleave: 2`），提高计算与通信的相互掩盖。
- **优化器并行**：优化器计算分散到多个设备上，以减少内存占用（`enable_parallel_optimizer: True`）。

> 开启细粒度多副本并行的同时必须开启序列并行。

通过以上配置，Llama3-70B的分布式训练在多机多卡环境中可以有效利用硬件资源，实现高效、稳定的模型训练。
