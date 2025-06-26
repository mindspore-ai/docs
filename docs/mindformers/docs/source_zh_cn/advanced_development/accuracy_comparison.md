# 与 Megatron-LM 比对训练精度

## 1. 概述

在大模型训练系统中，模型层级的数值精度验证是保障训练稳定性和结果可信度的关键环节。随着训练任务日益复杂，模型结构日趋庞大，确保不同实现之间在模型整体行为上的对齐，显得尤为重要。

Megatron-LM 是一个面向大规模训练任务的成熟框架，具备高度模块化与良好的可扩展性，广泛应用于高性能训练场景。MindSpore Transformers r1.6.0 版本在模型构建方面架构升级，以**ModuleSpec** 配置方式搭建模型，使得模型结构定义更加**灵活**且**易于复用**，极大提升了开发效率。同时在 NPU 环境下提供了全面优化的训练支持，能够充分发挥 NPU 架构优势。

本文档聚焦于两者在模型层面的训练精度一致性验证。通过构建等价的模型结构与配置，使用统一的输入，比较其前向输出、损失值、梯度行为等关键训练过程中的表现差异，以此验证 MindSpore Transformers 在 NPU 环境下实现的可靠性与精度可控性。

## 2. 环境说明

本节说明精度对比实验的推荐基础运行环境，包括：

### 驱动版本

| GPU  | 版本   | NPU  | 版本      |
|------|------|------|---------|
| CUDA | 12.1 | CANN | 8.1.RC1 |

### 重要库和依赖版本

| GPU                | 版本           | NPU                    | 版本      |
|--------------------|--------------|------------------------|---------|
| Megatron-LM        | core_r0.12.0 | MindSpore Transformers | dev     |
| Python             | \>=3.10      | Python                 | \>=3.10 |
| PyTorch            | 2.7.0        | MindSpore              | 2.6.0   |
| NumPy              | 1.26.4       | NumPy                  | 1.26.4  |
| Transformer Engine | 2.1.0        |                        |         |
| Apex               | 0.1          |                        |         |

### 镜像链接

上表中的 **GPU / NPU** 相关依赖版本为参考信息，实际环境请以对应官方镜像为准：

- **Megatron-LM**：参考 [Megatron-LM 文档](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.12.0?tab=readme-ov-file#setup)

- **MindSpore Transformers**：参考 [MindSpore Transformers 文档](https://gitee.com/mindspore/mindformers/blob/dev/README_CN.md)

## 3. 精度对比流程

本节介绍 MindSpore Transformers 在 NPU 环境下与业界主流实现 Megatron-LM 进行模型级别的精度对齐验证流程。本流程旨在指导用户完成从模型配置、数据输入、前向输出到梯度反向传播的全流程对齐，最终评估两个框架在相同任务下的精度一致性。

### 3.1 配置对齐

精度对比流程的第一步是确保两个框架使用**完全一致的模型配置**。为此，本小节提供了 [Megatron-LM](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/accuracy_comparison/example.sh) 与 [MindSpore Transformers] 的对应配置文件，分别定义了模型结构、并行策略以及关键训练超参数。

配置对齐的目标是保证两个系统在初始化状态下尽可能一致，从而使得后续的前向输出、梯度反向传播等比对具有可比性。

以 Megatron-LM 为主的配置的对照情况如下各表所示：

- 模型配置

    本文档仅支持 mcore 模型的精度比对，故 Megatron-LM 必须配置 `use-mcore-model`，MindSpore Transformers 必须配置`use_legacy: False`

    | Megatron-LM                                | 含义                                          | MindSpore Transformers                     | 含义                                                                  |
    |--------------------------------------------|---------------------------------------------|--------------------------------------------|---------------------------------------------------------------------|
    | `use-legacy-model`和`use-mcore-model`组合     | 是否使用 mcore 模型                               | `use_legacy`                               | 是否使用 mcore 模型                                                       |
    | `num-layers`                               | 网络层数，Transformer层的数量                        | `num_layers`                               | 网络层数，Transformer层的数量                                                |
    | `encoder-num-layers`                       | 编码器（Encoder）层数                              | 不支持配置                                      |                                                                     |
    | `decoder-num-layers`                       | 解码器（Decoder）层数                              | 不支持配置                                      |                                                                     |
    | `hidden-size`                              | 隐藏层大小，隐藏状态的维度                               | `hidden_size`                              | 隐藏层大小，隐藏状态的维度                                                       |
    | `ffn-hidden-size`                          | 前馈网络隐藏层大小                                   | `intermediate_size`                        | 前馈网络隐藏层大小                                                           |
    | `num-attention-heads`                      | 注意力头数                                       | `num_heads`                                | 注意力头数                                                               |
    | `kv-channels`                              | Key/Value 张量通道数                             | `head_dim`                                 | Key/Value 张量通道数                                                     |
    | `group-query-attention`                    | 是否启用分组查询注意力                                 | `use_gqa`                                  | 是否启用分组查询注意力                                                         |
    | `num-query-groups`                         | 查询分组数量                                      | `n_kv_heads`                               | 查询分组数量                                                              |
    | `max-position-embeddings`                  | 最大位置编码长度                                    | `max_position_embeddings`                  | 最大位置编码长度                                                            |
    | `position-embedding-type`                  | 位置编码类型，如 learned_absolute、rope 等            | `position_embedding_type`                  | 位置编码类型，如 learned_absolute、rope 等                                    |
    | `use-rotary-position-embeddings`           | 是否使用旋转位置编码（RoPE）                            | 由`position_embedding_type`==`rope`控制       | 是否使用旋转位置编码（RoPE）                                                    |
    | `rotary-base`                              | 旋转基数，用于 RoPE                                | `rotary_base`                              | 旋转基数，用于 RoPE                                                        |
    | `rotary-percent`                           | 旋转位置编码应用比例                                  | `rotary_percent`                           | 旋转位置编码应用比例                                                          |
    | `rotary-interleaved`                       | 是否使用交错的旋转编码                                 | `rotary_interleaved`                       | 是否使用交错的旋转编码                                                         |
    | `rotary-seq-len-interpolation-factor`      | 旋转序列长度插值因子                                  | `rotary_seq_len_interpolation_factor`      | 旋转序列长度插值因子                                                          |
    | `use-rope-scaling`                         | 是否启用 RoPE 缩放                                | `use_rope_scaling`                         | 是否启用 RoPE 缩放                                                        |
    | `rope-scaling-factor`                      | RoPE 缩放因子                                   | `no_position_embedding`                    | 是否禁用位置编码                                                            |
    | `no-position-embedding`                    | 是否禁用位置编码                                    | 不支持配置                                      | 默认不禁用                                                               |
    | `disable-bias-linear`                      | 不在线性层使用bias                                 | `add_bias_linear`                          | 在线性层使用 bias                                                         |
    | `mrope-section`                            | 多段 RoPE 段信息（多个段）                            | 不支持配置                                      |                                                                     |
    | `make-vocab-size-divisible-by`             | 使词表大小可被指定数整除                                | 不支持配置                                      | 默认不修改词表大小                                                           |
    | `init-method-std`                          | 模型参数初始化时使用的正态分布的标准差                         | `init_method_std`                          | 模型参数初始化时使用的正态分布的标准差                                                 |
    | `attention-dropout`                        | 多头自注意力机制里应用的 Dropout 概率                     | `attention_dropout`                        | 多头自注意力机制里应用的 Dropout 概率                                             |
    | `hidden-dropout`                           | 隐藏层的 Dropout 概率                             | `hidden_dropout`                           | 隐藏层的 Dropout 概率                                                     |
    | `normalization`                            | 归一化方法，LayerNorm 或 RMSNorm                   | `normalization`                            | 归一化方法，LayerNorm 或 RMSNorm                                           |
    | `norm-epsilon`                             | 归一化稳定因子（epsilon）                            | `rms_norm_eps`                             | RMSNorm 稳定因子                                                        |
    | `apply-layernorm-1p`                       | 是否在 LayerNorm 后应用 1 加法                      | 不支持配置                                      |                                                                     |
    | `apply-residual-connection-post-layernorm` | 残差连接是否在 LayerNorm 之后应用                      | `apply_residual_connection_post_layernorm` | 残差连接是否在 LayerNorm 之后应用                                              |
    | `openai-gelu`                              | 是否使用 OpenAI 版本的 GELU 激活函数                   | 不支持配置                                      |                                                                     |
    | `squared-relu`                             | 是否使用平方 ReLU 激活函数                            | 不支持配置                                      |                                                                     |
    | 由`swiglu`，`openai-gelu`，`squared-relu`控制   | 默认为 torch.nn.functional.gelu                | `hidden_act`                               | 激活函数类型                                                              |
    | `gated_linear_unit`                        | 多层感知机（MLP）中是否使用门控线性单元                       | `gated_linear_unit`                        | 多层感知机（MLP）中是否使用门控线性单元                                               |
    | `swiglu`                                   | 是否使用 SwiGLU 激活函数                            | `hidden_act`==`silu`和`gated_linear_unit`组合 | 是否使用 SwiGLU 激活函数                                                    |
    | `no-persist-layer-norm`                    | 禁用持久化层归一化                                   | 不支持配置                                      |                                                                     |
    | `untie-embeddings-and-output-weights`      | 是否解耦输入嵌入层和输出层权重                             | `untie_embeddings_and_output_weights`      | 是否解耦输入嵌入层和输出层权重                                                     |
    | 由`fp16` 和 `bf16` 控制                        | 训练中张量计算精度                                   | `compute_dtype`                            | 训练中张量计算精度                                                           |
    | `grad-reduce-in-bf16`                      | 以 BFloat16 执行梯度规约                           | 不支持配置                                      |                                                                     |
    | 不支持配置                                      | 默认以 BFloat16 生成初始化张量                        | `param_init_type`                          | 权重张量初始化精度，默认 Float32，以保证反向梯度以 Float32 更新                            |
    | 不支持配置                                      | 默认以 Float32 精度计算层归一化                        | `layernorm_compute_type`                   | 层归一化张量计算精度                                                          |
    | `attention-softmax-in-fp32`                | 在 Float32 中执行 attention softmax             | `softmax_compute_type`                     | softmax 张量计算精度                                                      |
    | 不支持配置                                      |                                             | `rotary_dtype`                             | 位置编码张量计算精度                                                          |
    | `loss-scale`                               | 总体损失缩放因子                                    | `loss_scale_value`                         | 总体损失缩放因子，配置在 runner_wrapper 中，`compute_dtype`为BFloat16的场景下，通常设置为1.0 |
    | `initial-loss-scale`                       | 初始损失缩放因子                                    | 不支持配置                                      |                                                                     |
    | `min-loss-scale`                           | 最小损失缩放因子                                    | 不支持配置                                      |                                                                     |
    | `loss-scale-window`                        | 动态缩放窗口大小                                    | `loss_scale_window`                        | 动态缩放窗口大小                                                            |
    | `hysteresis`                               | 损失缩放迟滞参数                                    | 不支持配置                                      |                                                                     |
    | `fp32-residual-connection`                 | 使用 Float32 残差连接                             | 不支持配置                                      |                                                                     |
    | `accumulate-allreduce-grads-in-fp32`       | 使用 Float32 累加并规约梯度                          | 不支持配置                                      | 默认使用 Float32 累加并规约梯度                                                |
    | `fp16-lm-cross-entropy`                    | 使用 Float16 执行语言模型交叉熵                        | 不支持配置                                      | 默认使用 Float32 执行语言模型交叉熵                                              |
    | `q-lora-rank`                              | Query 投影层的 LoRA rank，启用 Q-LoRA 时使用          | `q_lora_rank`                              | Query 投影层的 LoRA rank，启用 Q-LoRA 时使用                                  |
    | `kv-lora-rank`                             | Key/Value 投影层的 LoRA rank，启用 KV-LoRA 时使用     | `kv_lora_rank`                             | Key/Value 投影层的 LoRA rank，启用 KV-LoRA 时使用                             |
    | `qk-head-dim`                              | Q/K 每个头的维度（QK 头维度）                          | `qk_nope_head_dim`                         | Q/K 每个头的维度（QK 头维度）                                                  |
    | `qk-pos-emb-head-dim`                      | QK 相对位置嵌入的每头维度                              | `qk_rope_head_dim`                         | QK 相对位置嵌入的每头维度                                                      |
    | `v-head-dim`                               | Value 投影每头的维度（V 头维度）                        | `v_head_dim`                               | Value 投影每头的维度（V 头维度）                                                |
    | `rotary-scaling-factor`                    | Rotary Positional Embedding 缩放因子（RoPE 缩放系数） | `scaling_factor`                           | Rotary Positional Embedding 缩放因子（RoPE 缩放系数）                         |
    | `use-precision-aware-optimizer`            | 启用精度感知的优化器，用于自动管理不同 dtype 的参数更新             | 不支持配置                                      |                                                                     |
    | `main-grads-dtype`                         | 主梯度的数据类型                                    | 不支持配置                                      | 默认使用 Float32 作为主梯度的数据类型                                             |
    | `main-params-dtype`                        | 主参数的数据类型                                    | 不支持配置                                      | 默认使用 Float32 作为主参数的数据类型                                             |
    | `exp-avg-dtype`                            | EMA（指数移动平均）的数据类型                            | 不支持配置                                      |                                                                     |
    | `exp-avg-sq-dtype`                         | EMA平方项的数据类型                                 | 不支持配置                                      |                                                                     |
    | `first-last-layers-bf16`                   | 是否将首尾层强制使用 BFloat16                         | 不支持配置                                      |                                                                     |
    | `num-layers-at-start-in-bf16`              | 开始部分使用 BFloat16 的层数                         | 不支持配置                                      |                                                                     |
    | `num-layers-at-end-in-bf16`                | 末尾部分使用 BFloat16 的层数                         | 不支持配置                                      |                                                                     |
    | `multi-latent-attention`                   | 是否启用多隐变量注意力机制                               | `multi_latent_attention`                   | 是否启用多隐变量注意力机制                                                       |
    | `qk-layernorm`                             | 启用Query/Key 层归一化                            | `qk-layernorm`                             | 启用Query/Key 层归一化                                                    |

- 优化器与学习率调度配置

    | Megatron-LM               | 含义                                | MindSpore Transformers | 含义                                 |
    |---------------------------|-----------------------------------|------------------------|------------------------------------|
    | `optimizer`               | 优化器类型，如 adam、sgd 等                | `type`                 | 优化器类型，如 adam、sgd 等                 |
    | `adam-beta1`和`adam-beta2` | Adam 优化器的 β 参数                    | `betas`                | Adam 优化器的 β 参数                     |
    | `adam-eps`                | Adam 优化器中的 ε（防止除零）                | `eps`                  | Adam 优化器中的 ε（防止除零）                 |
    | `weight-decay`            | 权重衰减系数                            | `weight-decay`         | 权重衰减系数                             |
    | `start-weight-decay`      | 初始权重衰减值                           | 不支持配置                  |                                    |
    | `end-weight-decay`        | 最终权重衰减值                           | 不支持配置                  |                                    |
    | `weight-decay-incr-style` | 权重衰减调整策略，如 constant、linear、cosine | 不支持配置                  |                                    |
    | `clip-grad`               | 梯度裁剪阈值                            | `clip_grad`            | 梯度裁剪阈值，配置在 runner_wrapper 中，通常为1.0 |
    | `lr`                      | 学习率                               | `learning_rate`        | 学习率                                |
    | `lr-decay-style`          | 学习率衰减方式                           | `type`                 | 学习率衰减方式                            |
    | `lr-decay-iters`          | 学习率衰减对应迭代数                        | `total_steps`          | 默认为全部迭代数                           |
    | `lr-decay-samples`        | 学习率衰减对应样本数                        | 不支持配置                  |                                    |
    | `lr-warmup-iters`         | 学习率预热迭代步数                         | `warmup_steps`         | 学习率预热迭代步数                          |
    | `lr-warmup-fraction`      | 学习率预热阶段比例                         | `warmup_ratio`         | 学习率预热阶段比例                          |
    | `lr-warmup-init`          | 学习率预热起始值                          | `warmup_lr_init`       | 学习率预热起始值                           |
    | `min-lr`                  | 最小学习率                             | `min-lr`               | 最小学习率                              |

- 并行与分布式配置

    | Megatron-LM                            | 含义                                         | MindSpore Transformers              | 含义                        |
    |----------------------------------------|--------------------------------------------|-------------------------------------|---------------------------|
    | `tensor-model-parallel-size`           | 张量模型并行大小                                   | `model_parallel`                    | 张量模型并行大小                  |
    | `pipeline-model-parallel-size`         | 流水线模型并行大小                                  | `pipeline_stage`                    | 流水线模型并行大小                 |
    | `sequence-parallel`                    | 是否启用序列并行                                   | `use_seq_parallel`                  | 是否启用序列并行                  |
    | `context-parallel-size`                | 上下文并行大小                                    | `context_parallel`                  | 上下文并行大小                   |
    | `use-distributed-optimizer`            | 是否使用分布式优化器                                 | `parallel_optimizer_config`         | 是否使用分布式优化器                |
    | `expert-model-parallel-size`           | Expert 层模型并行度                              | `expert_parallel`                   | Expert 层模型并行度             |
    | `expert-tensor-parallel-size`          | Expert 层 tensor 并行度                        | `expert_model_parallel`             | Expert 层 tensor 并行度       |

- FlashAttention / Fused Attention 相关

    | Megatron-LM                 | 含义                                     | MindSpore Transformers | 含义                       |
    |-----------------------------|----------------------------------------|------------------------|--------------------------|
    | `attention-backend`         | 注意力实现后端：flash、fused、unfused、local、auto | 不支持配置                  |                          |
    | `use-flash-attn`            | 是否启用 FlashAttention                    | `use_flash_attention`  | 是否启用 FlashAttention，默认启用 |
    | `no-masked-softmax-fusion`  | 禁用 masked softmax 融合                   | 不支持配置                  |                          |
    | `no-bias-gelu-fusion`       | 禁用 bias + GELU 融合                      | 不支持配置                  |                          |
    | `no-bias-swiglu-fusion`     | 禁用 bias + SwiGLU 融合                    | 不支持配置                  |                          |
    | `no-bias-dropout-fusion`    | 禁用 bias + Dropout 融合                   | 不支持配置                  |                          |
    | `no-rope-fusion`            | 禁用 RoPE 融合                             | 不支持配置                  |                          |
    | `cross-entropy-loss-fusion` | 启用交叉熵损失融合                              | 不支持配置                  |                          |

- MoE 相关

    | Megatron-LM                           | 含义                         | MindSpore Transformers                | 含义                         |
    |---------------------------------------|----------------------------|---------------------------------------|----------------------------|
    | `num-experts`                         | 每层的专家数                     | `num-experts`                         | 每层的专家数                     |
    | `moe-layer-freq`                      | 每隔多少层插入 MoE 层              | `moe-layer-freq`                      | 每隔多少层插入 MoE 层              |
    | `moe-ffn-hidden-size`                 | MoE 中 FFN 隐藏层维度            | `moe_intermediate_size`               | MoE 中 FFN 隐藏层维度            |
    | `moe-shared-expert-intermediate-size` | 多专家共享中间维度大小                | `moe_shared_expert_intermediate_size` | 多专家共享中间维度大小                |
    | `moe-shared-expert-overlap`           | 是否重叠共享专家中间层                | `moe_shared_expert_overlap`           | 是否重叠共享专家中间层                |
    | `moe-grouped-gemm`                    | 是否使用 Grouped GEMM 优化       | `use_gmm`                             | 是否使用 Grouped GEMM 优化       |
    | `moe-router-load-balancing-type`      | Router 负载均衡策略              | `moe_router_load_balancing_type`      | Router 负载均衡策略              |
    | `moe-router-dtype`                    | Router 分数数据类型              | `router_dense_type`                   | Router 分数数据类型              |
    | `moe-router-score-function`           | Router 分数计算方式（如 softmax）   | `use_gating_sigmoid`                  | 是否应用 Sigmoid 激活函数          |
    | `moe-router-topk`                     | Router top-k 选择数目          | `num_experts_chosen`                  | Router top-k 选择数目          |
    | `moe-router-pre-softmax`              | 是否在 softmax 前进行处理          | `moe_router_pre_softmax`              | 是否在 softmax 前进行处理          |
    | `moe-router-num-groups`               | token 分组数                  | `n_groups`                            | token 分组数                  |
    | `moe-router-group-topk`               | 每组 token 的 top-k 数目        | `topk_group`                          | 每组 token 的 top-k 数目        |
    | `moe-router-topk-scaling-factor`      | top-k 分数缩放因子               | `routed_scaling_factor`               | top-k 分数缩放因子               |
    | `moe-router-enable-expert-bias`       | 是否使用 expert 的 bias         | `balance_via_topk_bias`               | 是否使用 expert 的 bias         |
    | `moe-router-bias-update-rate`         | expert bias 更新率            | `topk_bias_update_rate`               | expert bias 更新率            |
    | `moe-use-legacy-grouped-gemm`         | 是否使用旧版 Grouped GEMM        | 不支持配置                                 |                            |
    | `moe-aux-loss-coeff`                  | MoE 辅助损失系数                 | 不支持配置                                 |                            |
    | `moe-z-loss-coeff`                    | MoE z-loss 系数              | 不支持配置                                 |                            |
    | `moe-input-jitter-eps`                | MoE 输入 jitter 噪声量          | `moe_input_jitter_eps`                | MoE 输入 jitter 噪声量          |
    | `moe-token-dispatcher-type`           | token 调度策略（allgather 等）    | 不支持配置                                 |                            |
    | `moe-enable-deepep`                   | 是否启用 DeepEP 混合专家优化         | `moe_enable_deepep`                   | 是否启用 DeepEP 混合专家优化         |
    | `moe-per-layer-logging`               | 每层 MoE 打印日志                | `moe_per_layer_logging`               | 每层 MoE 打印日志                |
    | `moe-expert-capacity-factor`          | expert 容量扩展比例              | `capacity_factor`                     | expert 容量扩展比例              |
    | `moe-pad-expert-input-to-capacity`    | 是否填充 expert 输入到容量上限        | `moe_pad_expert_input_to_capacity`    | 是否填充 expert 输入到容量上限        |
    | `moe-token-drop-policy`               | token 丢弃策略（probs/position） | `enable_sdrop`                        | token 丢弃策略（probs/position） |
    | `moe-extended-tp`                     | 启用扩展 tensor 并行支持           | 不支持配置                                 |                            |
    | `moe-use-upcycling`                   | 是否启用专家 upcycling           | 不支持配置                                 |                            |
    | `moe-permute-fusion`                  | 启用专家内部 permute 融合优化        | `moe_permute_fusion`                  | 启用专家内部 permute 融合优化        |
    | `mtp-num-layers`                      | MoE 层的数量                   | `mtp_depth`                           | MoE 层的数量                   |
    | `mtp-loss-scaling-factor`             | MoE 架构中的损失缩放               | `mtp_loss_factor`                     | MoE 架构中的损失缩放               |

- 数据加载与分词设置

    | Megatron-LM                   | 含义                        | MindSpore Transformers | 含义                             |
    |-------------------------------|---------------------------|------------------------|--------------------------------|
    | `data-path`和`split`组合         | 通用数据路径                    | `data_path`            | Megatron数据集采样比例以及路径            |
    | `train-data-path`             | 训练数据路径                    | 不支持配置                  |                                |
    | `valid-data-path`             | 验证数据路径                    | 不支持配置                  |                                |
    | `test-data-path`              | 测试数据路径                    | 不支持配置                  |                                |
    | `vocab-size`                  | 词表大小                      | `vocab_size`           | 词表大小                           |
    | `vocab-file`                  | 词表文件路径                    | 不支持配置                  |                                |
    | `merge-file`                  | BPE 合并规则文件                | 不支持配置                  |                                |
    | `tokenizer-type`              | 分词器类型（如 GPT2BPETokenizer） | 不支持配置                  | 默认使用 Huggingface 对应的 Tokenizer |
    | `seq-length`                  | 输入序列长度                    | `seq_length`           | 输入序列长度                         |
    | `encoder-seq-length`          | 编码器输入长度                   | 不支持配置                  |                                |
    | `decoder-seq-length`          | 解码器输入长度                   | 不支持配置                  |                                |
    | `retriever-seq-length`        | 检索器序列长度（如果启用）             | 不支持配置                  |                                |
    | `num-workers`                 | 加载数据的线程数                  | `num_parallel_workers` | 加载数据的线程数                       |
    | `num-dataset-builder-threads` | 构建数据集的线程数                 | 不支持配置                  |                                |
    | `data-cache-path`             | 数据缓存路径                    | 不支持配置                  |                                |

- 训练控制与保存

    | Megatron-LM                    | 含义                     | MindSpore Transformers                 | 含义                                                                          |
    |--------------------------------|------------------------|----------------------------------------|-----------------------------------------------------------------------------|
    | 不支持配置                          | 每个迭代处理的局部样本总数          | `batch_size`                           | 每个迭代处理的局部样本总数，在`runner_wrapper`中配置                                          |
    | 不支持配置                          | 每个迭代处理的局部样本总数          | `micro_batch_interleave_num`           | 微批交错数，当`micro_batch_interleave_num`大于 1 时，启用多副本并行                           |
    | `global_batch_size`            | 每个迭代处理的全局样本总数          | `batch_size`和`data_parallel`组合         | 每个迭代处理的全局样本总数，`batch_size`，`data_parallel`和`micro_batch_interleave_num`相乘得到 |
    | 不支持配置                          | 迭代周期数                  | `epochs`                               | 迭代周期数，在`runner_wrapper`中配置                                                  |
    | `train-samples`                | 总训练样本数                 | `sizes`                                | 总训练样本数，在`train_dataset`中配置                                                  |
    | `train-iters`                  | 总训练迭代次数                | `epochs`，`sizes`和`global_batch_size`组合 | 总训练迭代次数，`sizes`除`global_batch_size`再乘`epochs`得到                             |
    | `log-interval`                 | 日志记录间隔（迭代步数）           | `per_print_times`                      | 日志记录间隔（迭代步数），在`callbacks`的`MFLossMonitor`中配置                                |
    | `eval-iters`                   | 每次评估时使用的迭代步数           | 不支持配置                                  |                                                                             |
    | `eval-interval`                | 评估间隔步数                 | 不支持配置                                  |                                                                             |
    | `save`                         | 模型保存路径                 | `output_dir`                           | 模型保存路径                                                                      |
    | `save-interval`                | 模型保存间隔（迭代步数）           | `save_checkpoint_steps`                | 模型保存间隔（迭代步数），在`callbacks`的`CheckpointMonitor`中配置                            |
    | `non-persistent-save-interval` | 临时保存间隔（非持久化）           | 不支持配置                                  |                                                                             |
    | `non-persistent-ckpt-type`     | 临时保存类型（如 global/local） | 不支持配置                                  |                                                                             |
    | `pretrained-checkpoint`        | 预训练模型路径                | 不支持配置                                  |                                                                             |
    | `ckpt-step`                    | 加载指定 step 的权重          | `load_checkpoint`和`resume_training`组合  | 断点续训场景下，加载指定名字的权重                                                           |
    | `load`                         | 从该路径加载模型               | `load_checkpoint`                      | 从该路径加载模型                                                                    |
    | `exit-interval`                | 控制退出训练的迭代间隔            | `stop_step`                            | 控制退出训练的迭代数，在`callbacks`的`TrainCallMonitor`中配置                               |
    | `exit-duration-in-mins`        | 控制退出训练的时间限制（分钟）        | 不支持配置                                  |                                                                             |

- 重计算配置

    MindSpore Transformers 重计算配置逻辑与 Megatron-LM 差异较大，参考[重计算配置](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/memory_optimization.html#%E9%87%8D%E8%AE%A1%E7%AE%97)使能即可。

    | Megatron-LM                    | 含义                    | MindSpore Transformers | 含义                       |
    |--------------------------------|-----------------------|------------------------|--------------------------|
    | `recompute-activations`        | 是否启用激活重计算以节省内存        | `recompute`            | 是否启用激活完全重计算以节省内存（`bool`） |
    | `recompute-granularity`        | 重计算粒度（full/selective） | `select_recompute`     | 是否开启 selective 重计算       |
    | `recompute-method`             | 重计算方法（uniform/block）  | 不支持配置                  |                          |
    | `recompute-num-layers`         | 重计算的层数                | `recompute`            | 重计算的层数（`tuple`/`list`）   |
    | `distribute-saved-activations` | 分布式存储激活值              | 不支持配置                  |                          |
    | `checkpoint-activations`       | 是否启用激活值检查点机制以减少显存     | 不支持配置                  |                          |
    | `moe-layer-recompute`          | MoE 层启用重计算            | 不支持配置                  |                          |

**注意**：两个框架还有其他训练相关性较小的配置，MindSpore Transformer 详情参考[配置说明]，Megatron-LM 可通过执行命令`torchrun --nproc_per_node=1 pretrain_gpt.py --help`查看。

### 3.2 数据集对齐

精度对比流程中，必须确保两个框架使用完全一致的数据输入。该小节将介绍如何对齐 Megatron-LM 与 MindSpore Transformers 的数据集制作和配置，从而保证输入样本的一致性，为后续权重加载与精度验证提供基础。

#### 3.2.1 数据集准备

两个框架均支持加载 Megatron 数据集，该数据集通常经过预处理，序列化为二进制格式（例如`.bin`或`.idx`文件），并配套特定索引机制，便于在分布式集群环境下高效并行加载与数据切分。

- 数据集下载：[wikitext-103数据集](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens)

- 分词模型下载：分词模型[tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json?download=true)

#### 3.2.2 数据集处理

- 生成Megatron BIN格式文件

   将数据集文件`wiki.train.tokens`和分词模型文件`tokenizer.json`放置在`../dataset`下，并参照[Megatron数据集-数据预处理](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/dataset.html#%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86)制作`data.json`文件。

   使用以下命令将数据集文件转换为BIN格式文件。

   ```shell
   cd $MINDFORMERS_HOME
   python mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py \
    --input /path/data.json \
    --output-prefix ../dataset/wiki_4096 \
    --vocab-file ../dataset/tokenizer.json \
    --seq-length 4096 \
    --workers 1
   ```

- 构建Megatron BIN数据集模块

   执行如下命令构建Megatron BIN数据集模块。

   ```shell
   pip install pybind11
   cd $MINDFORMERS_HOME/mindformers/dataset/blended_datasets
   make
   ```

   其中，`$MINDFORMERS_HOME` 指 Mindspore Transformers 源代码所在的目录。

#### 3.2.2 数据集配置

本小节会将两个框架配置文件中的数据集配置项，进行对比和说明。

- Megatron-LM:

    Megatron-LM 样例中的数据集配置项如下：

    ```shell
    TOKENIZER_MODEL="/path/to/tokenizer.json"
    DATA_PATH="/path/to/wiki_text_document"

    DATA_ARGS=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model ${TOKENIZER_MODEL}
        --data-path $DATA_PATH
        --split 1,0,0
    )
    ```

    其中，

    - `tokenizer-type`为分词模型文件类型
    - `tokenizer-model`为分词模型文件`tokenizer.json`的所在位置，精确到完整文件名
    - `data-path`为处理好的数据集的所在位置，精确到`.bin`或`.idx`文件的前缀
    - `split`为数据集的采样比例

- MindSpore Transformers:

    MindSpore Transformers 样例中相对应的数据集配置项如下:

    ```yaml
    config:  # GPTDataset配置项
      data_path:  # Megatron数据集采样比例以及路径
        - '1'
        - "/home/to/wiki_text_document"
    ```

    其中，需要注意的是`data_path`的第一个参数是数据集采样比例，样例中的设置等价于 Megatron-LM 样例中的 `--split`；第二个参数是处理好的数据集的所在位置，精确到`.bin`或`.idx`文件的前缀，样例中的设置等价于 Megatron-LM 样例中的 `--data-path`

### 3.3 权重对齐

为了实现不同框架间模型行为的一致性，需将训练得到的权重精确映射到 MindSpore Transformers 和 Megatron-LM 中对应位置，通过合理的权重转换和切分实现。

#### 权重转换

由于 MindSpore Transformers 和 Megatron-LM 使用的权重格式、参数命名方式及张量排列存在差异，直接加载权重通常会导致不兼容。因此，需要通过专门的转换脚本将源框架导出的模型权重转换为目标框架可识别的格式。

1. 生成 MinSpore Transformers 初始权重

   参照[callbacks 配置](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/configuration.html#callbacks%E9%85%8D%E7%BD%AE)通过修改 `example.yaml` 文件并执行[查看结果](#34-查看结果)中提供的命令，即可通过预训练在`example.yaml`中的`output_dir`的`checkpoints`下获得一份初始权重，修改内容如下：

   ```yaml
   # Before (example.yaml)
   load_checkpoint: '/path/to/checkpoints/'
   ```

   ```yaml
   # After (example.yaml)
   load_checkpoint: ''

   callbacks:
   - type: CheckpointMonitor
     prefix: "deepseekv3"
     save_checkpoint_steps: 1
     keep_checkpoint_max: 2
     integrated_save: False
     async_save: False
     checkpoint_format: "safetensors"
   - type: TrainCallBack
     stop_step: 1
   ```

   **注意**：获得权重之后，需要将`example.yaml`反向修改复原。

2. MindSpore Transformers to Megatron-LM

   为了将 MindSpore Transformers 的权重精确映射为 Megatron-LM 可加载的等价权重，我们将会提供转换权重脚本，执行权重转换脚本即可获得等价权重。

### 3.4 查看结果

完成以上步骤后，即可进行训练，从日志中输出的结果中提取关键数据查看精度对比结果。

- Megatron-LM

  将`example.sh`文件放到 Megatron-LM 代码目录下，执行以下代码：  

  ```shell
  bash example.sh
  ```

- MindSpore Transformers

  在 MindSpore Transformer 代码目录下，执行以下代码：

  ```shell
  bash scripts/msrun_launcher.sh "run_mindformer.py \
   --config /path/to/example.yaml"
  ```

  其中，`config`是模型的配置文件，文件在 MindSpore Transformers 代码仓中 config 目录下

- 结果对比

  分别查看二者的输出日志，Megatron-LM 的日志位置为`example.sh`中的`logs/${logtime}.log`, MindSpore Transformer 的日志位置为`example.yaml`中的`output_dir`的`msrun_log/worker_0.log`。结果对比参考下表：

  | Megatron-LM     | MindSpore Transformers | 含义                                                                                                                                                             |
  |-----------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | `iteration`     | `epoch` 与 `step` 的组合   | 表示训练过程中的全局迭代次数。MindSpore Transformers 通常以 `(epoch, step)` 表示当前训练位置，而 Megatron-LM 使用单一的 `iteration` 表示。两者关系为：`iteration = (epoch - 1) * steps_per_epoch + step` |
  | `lm loss`       | `loss`                 | 训练损失，精度对比核心指标。MindSpore Transformers 的`loss`是指`lm loss`和`aux loss`的和，未来将会分别打印输出                                                                                |
  | `learning rate` | `lr`                   | 学习率，精度对比参考指标                                                                                                                                                   |
  | `grand norm`    | `global norm`          | 全局梯度范数，精度对比参考指标                                                                                                                                                |