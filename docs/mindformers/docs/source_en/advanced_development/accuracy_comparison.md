# Comparing the Model Precision with that of Megatron-LM

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/advanced_development/accuracy_comparison.md)

## 1. Overview

In the LLM training system, precision validation at the model level is a key step to ensure training stability and result reliability. As training tasks become increasingly complex and model structures become larger, it is particularly important to ensure alignment of the overall model behavior between different implementations.

Megatron-LM is a mature framework for large-scale training tasks. It is highly modular and scalable and is widely used in training scenarios with high performance requirements. MindSpore Transformers r1.6.0 upgrades the model architecture by using the ModuleSpec configuration mode to build models. This makes model structure definition **more flexible** and **easier to reuse**, greatly improving development efficiency. In addition, comprehensive training support is provided in the NPU environment, fully leveraging the advantages of the NPU architecture.

This document focuses on the validation of precision consistency at the model level. By building equivalent model structures and configurations and using unified inputs, this document compares key training performance indicators such as the forward output, loss, and gradient behavior to validate the reliability and precision controllability of MindSpore Transformers in the NPU environment.

## 2. Environment

This section describes the recommended basic operating environment for the precision comparison experiment.

### Driver Version

| GPU  | Version  | NPU  | Version     |
|------|------|------|---------|
| CUDA | 12.1 | CANN | 8.1.RC1 |

### Important Libraries and Dependency Versions

| GPU                | Version          | NPU                    | Version     |
|--------------------|--------------|------------------------|---------|
| Megatron-LM        | core_r0.12.0 | MindSpore Transformers | master     |
| Python             | 3.10 or later     | Python                 | 3.10 or later|
| PyTorch            | 2.7.0        | MindSpore              | 2.6.0   |
| NumPy              | 1.26.4       | NumPy                  | 1.26.4  |
| Transformer Engine | 2.1.0        |                        |         |
| Apex               | 0.1          |                        |         |

### Image Links

The **GPU/NPU** dependency versions in the preceding tables are for reference only. The actual versions in official images prevail.

- **Megatron-LM**: For details, see [Megatron-LM documentation](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.12.0?tab=readme-ov-file#setup).

- **MindSpore Transformers**: For details, see [MindSpore Transformers documentation](https://gitee.com/mindspore/mindformers/blob/master/README.md).

## 3. Precision Comparison Process

This section describes the model-level precision consistency validation process between MindSpore Transformers and the mainstream Megatron-LM in an NPU environment. This process is used to guide users through the entire alignment from model configuration, data input, and forward output, to gradient backpropagation, and finally evaluate the precision consistency of the two frameworks under the same task.

### 3.1 Configuration Alignment

The first step of the precision comparison process is to ensure that the two frameworks use **the same model configuration**. This section provides the configuration files of [Megatron-LM](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/accuracy_comparison/example.sh) and [MindSpore Transformers](https://gitee.com/mindspore/mindformers), which define the model structure, parallel policy, and key training hyperparameters.

The configuration alignment aims to ensure that the two systems are as consistent as possible in the initial state, so that the forward output and gradient backpropagation can be compared.

The following tables describe the configuration comparison with Megatron-LM.

- Model configurations

    This document supports only the precision comparison of the mcore model. Therefore, `--use-mcore-model` must be configured for Megatron-LM, and `use_legacy: False` must be configured for MindSpore Transformers.

    | Megatron-LM                                | Description                                         | MindSpore Transformers                     | Description                                                                                                                                             |
    |--------------------------------------------|---------------------------------------------|--------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
    | `use-legacy-model` and `use-mcore-model`    | Specifies whether to use the mcore model.                              | `use_legacy`                               | Specifies whether to use the mcore model. `use_legacy: False` is equivalent to `--use-mcore-model`.                                                     |
    | `num-layers`                               | Number of network layers, that is, number of transformer layers.                       | `num_layers`                               | Number of network layers, that is, number of transformer layers.                                                                                        |
    | `encoder-num-layers`                       | Number of encoder layers.                             | Not supported.                                     |                                                                                                                                                         |
    | `decoder-num-layers`                       | Number of decoder layers.                             | Not supported.                                     |                                                                                                                                                         |
    | `hidden-size`                              | Size of the hidden layer, which is the dimension in the hidden state.                              | `hidden_size`                              | Size of the hidden layer, which is the dimension in the hidden state.                                                                                   |
    | `ffn-hidden-size`                          | Size of the hidden layer in the feedforward network.                                  | `intermediate_size`                        | Size of the hidden layer in the feedforward network.                                                                                                    |
    | `num-attention-heads`                      | Number of attention heads.                                      | `num_heads`                                | Number of attention heads.                                                                                                                              |
    | `kv-channels`                              | Number of key/value tensor channels.                            | `head_dim`                                 | Number of key/value tensor channels.                                                                                                                    |
    | `group-query-attention`                    | Specifies whether to enable group query attention.                                | `use_gqa`                                  | Specifies whether to enable group query attention.                                                                                                      |
    | `num-query-groups`                         | Number of query groups.                                     | `n_kv_heads`                               | Number of query groups.                                                                                                                                 |
    | `max-position-embeddings`                  | Maximum position encoding length.                                   | `max_position_embeddings`                  | Maximum position encoding length.                                                                                                                       |
    | `position-embedding-type`                  | Position encoding type, such as learned_absolute and rope.           | `position_embedding_type`                  | Position encoding type, such as learned_absolute and rope.                                                                                              |
    | `use-rotary-position-embeddings`           | Specifies whether to use rotary position embedding (RoPE).                           | Specified by `position_embedding_type`==`rope`      | Specifies whether to use RoPE.                                                                                                                          |
    | `rotary-base`                              | Rotary base used for RoPE.                               | `rotary_base`                              | Rotary base used for RoPE.                                                                                                                              |
    | `rotary-percent`                           | RoPE usage ratio.                                 | `rotary_percent`                           | RoPE usage ratio.                                                                                                                                       |
    | `rotary-interleaved`                       | Specifies whether to use interleaved RoPE.                                | `rotary_interleaved`                       | Specifies whether to use interleaved RoPE.                                                                                                              |
    | `rotary-seq-len-interpolation-factor`      | Rotary sequence length interpolation factor.                                 | `rotary_seq_len_interpolation_factor`      | Rotary sequence length interpolation factor.                                                                                                            |
    | `use-rope-scaling`                         | Specifies whether to enable RoPE scaling.                               | `use_rope_scaling`                         | Specifies whether to enable RoPE scaling.                                                                                                               |
    | `rope-scaling-factor`                      | RoPE scaling factor.                                  | `scaling_factor`                    | RoPE scaling factor.                                                                                                                                    |
    | `no-position-embedding`                    | Specifies whether to disable location encoding.                                   | `no-position-embedding`                                     | Specifies whether to disable location encoding.                                                                                                         |
    | `disable-bias-linear`                      | Disables bias in linear layers.                                | `add_bias_linear`                          | Enables bias in linear layers.                                                                                                                          |
    | `mrope-section`                            | Information of multiple RoPE sections.                           | Not supported.                                     |                                                                                                                                                         |
    | `make-vocab-size-divisible-by`             | Divides the size of the word table by a specified number.                               | Not supported.                                     | By default, the dictionary size is not changed.                                                                                                         |
    | `init-method-std`                          | Standard deviation of the normal distribution used during model parameter initialization.                        | `init_method_std`                          | Standard deviation of the normal distribution used during model parameter initialization.                                                               |
    | `attention-dropout`                        | Dropout probability applied in the multi-head self-attention mechanism.                    | `attention_dropout`                        | Dropout probability applied in the multi-head self-attention mechanism.                                                                                 |
    | `hidden-dropout`                           | Dropout probability in the hidden layer.                            | `hidden_dropout`                           | Dropout probability in the hidden layer.                                                                                                                |
    | `normalization`                            | Normalization method, which can be LayerNorm or RMSNorm.                  | `normalization`                            | Normalization method, which can be LayerNorm or RMSNorm.                                                                                                |
    | `norm-epsilon`                             | Normalized stability factor (epsilon).                           | `rms_norm_eps`                             | RMSNorm stability factor.                                                                                                                               |
    | `apply-layernorm-1p`                       | Specifies whether to add 1 after LayerNorm.                     | Not supported.                                     |                                                                                                                                                         |
    | `apply-residual-connection-post-layernorm` | Specifies whether the residual connection is applied after LayerNorm.                     | `apply_residual_connection_post_layernorm` | Specifies whether the residual connection is applied after LayerNorm.                                                                                   |
    | `openai-gelu`                              | Specifies whether to use the GELU activation function of the OpenAI version.                  | Not supported.                                     |                                                                                                                                                         |
    | `squared-relu`                             | Specifies whether to use the square ReLU activation function.                           | Not supported.                                     |                                                                                                                                                         |
    | Specified by `swiglu`, `openai-gelu`, and `squared-relu`  | The default value is **torch.nn.functional.gelu**.               | `hidden_act`                               | Activation function type.                                                                                                                               |
    | `gated_linear_unit`                        | Specifies whether to use gate linear unit in multi-layer perceptron (MLP).                      | `gated_linear_unit`                        | Specifies whether to use gate linear unit in MLP.                                                                                                       |
    | `swiglu`                                   | Specifies whether to use the SwiGLU activation function.                           | `hidden_act`==`silu` and `gated_linear_unit`| Specifies whether to use the SwiGLU activation function.                                                                                                |
    | `no-persist-layer-norm`                    | Disables persistence layer normalization.                                  | Not supported.                                     |                                                                                                                                                         |
    | `untie-embeddings-and-output-weights`      | Specifies whether to decouple the weights of the input embedding layer and output layer.                            | `untie_embeddings_and_output_weights`      | Specifies whether to decouple the weights of the input embedding layer and output layer.                                                                |
    | Specified by `fp16` and `bf16`                       | Tensor compute precision during training.                                  | `compute_dtype`                            | Tensor compute precision during training.                                                                                                               |
    | `grad-reduce-in-bf16`                      | Gradient reduction using BFloat16.                          | Not supported.                                     |                                                                                                                                                         |
    | Not supported.                                     | By default, the initialization tensor is generated in BFloat16 format.                       | `param_init_type`                          | Initial precision of the weight tensor. The default value is **Float32**, which ensures that the backward gradient is updated in Float32.               |
    | Not supported.                                     | By default, layer normalization is calculated in Float32.                       | `layernorm_compute_type`                   | Layer normalization tensor calculation precision.                                                                                                       |
    | `attention-softmax-in-fp32`                | Executes **attention softmax** in Float32.            | `softmax_compute_type`                     | Softmax tensor calculation precision.                                                                                                                   |
    | Not supported.                                     |                                             | `rotary_dtype`                             | Position encoding tensor calculation precision.                                                                                                         |
    | `loss-scale`                               | Overall loss scaling factor.                                   | `loss_scale_value`                         | Overall loss scaling factor, which is configured in **runner_wrapper**. If `compute_dtype` is set to **BFloat16**, the value is usually set to **1.0**. |
    | `initial-loss-scale`                       | Initial loss scaling factor.                                   | Not supported.                                     |                                                                                                                                                         |
    | `min-loss-scale`                           | Minimum loss scaling factor.                                   | Not supported.                                     |                                                                                                                                                         |
    | `loss-scale-window`                        | Dynamic window size scaling.                                   | `loss_scale_window`                        | Dynamic window size scaling.                                                                                                                            |
    | `hysteresis`                               | Loss scale hysteresis parameter.                                   | Not supported.                                     |                                                                                                                                                         |
    | `fp32-residual-connection`                 | Uses Float32 for residual connection.                            | Not supported.                                     |                                                                                                                                                         |
    | `accumulate-allreduce-grads-in-fp32`       | Accumulates and reduces gradients using Float32.                         | Not supported.                                     | Accumulates and reduces gradients using Float32 by default.                                                                                             |
    | `fp16-lm-cross-entropy`                    | Uses Float16 to execute the cross entropy of the LLM.                       | Not supported.                                     | Uses Float32 to execute the cross entropy of the LLM by default.                                                                                        |
    | `q-lora-rank`                              | LoRA rank of the query projection layer, which is used when Q-LoRA is enabled.         | `q_lora_rank`                              | LoRA rank of the query projection layer, which is used when Q-LoRA is enabled.                                                                          |
    | `kv-lora-rank`                             | LoRA rank of the key/value projection layer, which is used when KV-LoRA is enabled.    | `kv_lora_rank`                             | LoRA rank of the key/value projection layer, which is used when KV-LoRA is enabled.                                                                     |
    | `qk-head-dim`                              | Number of dimensions per Q/K head.                         | `qk_nope_head_dim`                         | Number of dimensions per Q/K head.                                                                                                                      |
    | `qk-pos-emb-head-dim`                      | Number of relative position embedding dimensions per Q/K head.                             | `qk_rope_head_dim`                         | Number of relative position embedding dimensions per Q/K head.                                                                                          |
    | `v-head-dim`                               | Number of dimensions per value projection (V head).                       | `v_head_dim`                               | Number of dimensions per value projection (V head).                                                                                                     |
    | `rotary-scaling-factor`                    | RoPE scaling coefficient.| `scaling_factor`                           | RoPE scaling coefficient.                                                                                                                               |
    | `use-precision-aware-optimizer`            | Enables the optimizer with precision awareness to automatically manage parameter updates of different data types.            | Not supported.                                     |                                                                                                                                                         |
    | `main-grads-dtype`                         | Data type of the main gradient.                                   | Not supported.                                     | By default, Float32 is used as the data type of the main gradient.                                                                                      |
    | `main-params-dtype`                        | Data type of the main parameter.                                   | Not supported.                                     | By default, Float32 is used as the data type of the main parameter.                                                                                     |
    | `exp-avg-dtype`                            | Data type of the exponential moving average (EMA).                           | Not supported.                                     |                                                                                                                                                         |
    | `exp-avg-sq-dtype`                         | Data type of the EMA square item.                                | Not supported.                                     |                                                                                                                                                         |
    | `first-last-layers-bf16`                   | Specifies whether to forcibly use BFloat16 at the first and last layers.                        | Not supported.                                     |                                                                                                                                                         |
    | `num-layers-at-start-in-bf16`              | Number of layers that start with BFloat16.                        | Not supported.                                     |                                                                                                                                                         |
    | `num-layers-at-end-in-bf16`                | Number of layers that end with BFloat16.                        | Not supported.                                     |                                                                                                                                                         |
    | `multi-latent-attention`                   | Specifies whether to enable the multi-hidden variable attention mechanism.                              | `multi_latent_attention`                   | Specifies whether to enable the multi-hidden variable attention mechanism.                                                                              |
    | `qk-layernorm`                             | Enables query/key layer normalization.                           | `qk-layernorm`                             | Enables query/key layer normalization.                                                                                                                  |

- Optimizer and learning rate scheduling configurations

    | Megatron-LM               | Description                               | MindSpore Transformers | Description                                |
    |---------------------------|-----------------------------------|------------------------|------------------------------------|
    | `optimizer`               | Optimizer type, such as Adam and SGD.               | `type`                 | Optimizer type, such as Adam and SGD.                |
    | `adam-beta1` and `adam-beta2`| β parameter of the Adam optimizer.                   | `betas`                | β parameter of the Adam optimizer.                    |
    | `adam-eps`                | ε in the Adam optimizer (to prevent division by zero).               | `eps`                  | ε in the Adam optimizer (to prevent division by zero).                |
    | `weight-decay`            | Weight decay coefficient.                           | `weight-decay`         | Weight decay coefficient.                            |
    | `start-weight-decay`      | Initial weight decay.                          | Not supported.                 |                                    |
    | `end-weight-decay`        | Final weight decay.                          | Not supported.                 |                                    |
    | `weight-decay-incr-style` | Weight decay adjustment policy, which can be **constant**, **linear**, and **cosine**.| Not supported.                 |                                    |
    | `clip-grad`               | Gradient clipping threshold.                           | `clip_grad`            | Gradient clipping threshold, which is configured in **runner_wrapper**. The value is usually **1.0**.|
    | `lr`                      | Learning rate.                              | `learning_rate`        | Learning rate.                               |
    | `lr-decay-style`          | Learning rate decay mode.                          | `type`                 | Learning rate decay mode.                           |
    | `lr-decay-iters`          | Number of iterations corresponding to the learning rate decay.                       | `total_steps`          | Total number of iterations by default.                          |
    | `lr-decay-samples`        | Number of samples corresponding to the learning rate decay.                       | Not supported.                 |                                    |
    | `lr-warmup-iters`         | Number of warm-up iteration steps of the learning rate.                        | `warmup_steps`         | Number of warm-up iteration steps of the learning rate.                         |
    | `lr-warmup-fraction`      | Proportion of the learning rate warm-up phase.                        | `warmup_ratio`         | Proportion of the learning rate warm-up phase.                         |
    | `lr-warmup-init`          | Initial learning rate for warm-up.                         | `warmup_lr_init`       | Initial learning rate for warm-up.                          |
    | `min-lr`                  | Minimum learning rate.                            | `min-lr`               | Minimum learning rate.                             |

- Parallel and distributed configurations

    | Megatron-LM                            | Description                                        | MindSpore Transformers              | Description                       |
    |----------------------------------------|--------------------------------------------|-------------------------------------|---------------------------|
    | `tensor-model-parallel-size`           | Degree of tensor model parallelism.                                  | `model_parallel`                    | Degree of tensor model parallelism.                 |
    | `pipeline-model-parallel-size`         | Parallel size of the pipeline model.                                 | `pipeline_stage`                    | Parallel size of the pipeline model.                |
    | `sequence-parallel`                    | Specifies whether to enable sequence parallelism.                                  | `use_seq_parallel`                  | Specifies whether to enable sequence parallelism.                 |
    | `context-parallel-size`                | Context parallel size.                                   | `context_parallel`                  | Context parallel size.                  |
    | `use-distributed-optimizer`            | Specifies whether to use a distributed optimizer.                                | `parallel_optimizer_config`         | Specifies whether to use a distributed optimizer.               |
    | `expert-model-parallel-size`           | Degree of model parallelism at the expert layer.                             | `expert_parallel`                   | Degree of model parallelism at the expert layer.            |
    | `expert-tensor-parallel-size`          | Degree of tensor parallelism at the expert layer.                       | `expert_model_parallel`             | Degree of tensor parallelism at the expert layer.      |

- FlashAttention/Fused Attention

    | Megatron-LM                 | Description                                    | MindSpore Transformers | Description                      |
    |-----------------------------|----------------------------------------|------------------------|--------------------------|
    | `attention-backend`         | Attention implementation backend, which can be **flash**, **fused**, **unfused**, **local**, and **auto**.| Not supported.                 |                          |
    | `use-flash-attn`            | Specifies whether to enable FlashAttention.                   | `use_flash_attention`  | Specifies whether to enable FlashAttention. FlashAttention is enabled by default.|
    | `no-masked-softmax-fusion`  | Disables masked softmax fusion.                  | Not supported.                 |                          |
    | `no-bias-gelu-fusion`       | Disables bias+GELU fusion.                     | Not supported.                 |                          |
    | `no-bias-swiglu-fusion`     | Disables bias+SwiGLU fusion.                   | Not supported.                 |                          |
    | `no-bias-dropout-fusion`    | Disables bias+Dropout fusion.                  | Not supported.                 |                          |
    | `no-rope-fusion`            | Disables RoPE fusion.                            | Not supported.                 |                          |
    | `cross-entropy-loss-fusion` | Enables cross entropy loss fusion.                             | Not supported.                 |                          |

- MoE

    | Megatron-LM                           | Description                        | MindSpore Transformers                | Description                        |
    |---------------------------------------|----------------------------|---------------------------------------|----------------------------|
    | `num-experts`                         | Number of experts at each layer.                    | `num-experts`                         | Number of experts at each layer.                    |
    | `moe-layer-freq`                      | Number of layers between inserted MoE layers.             | `moe-layer-freq`                      | Number of layers between inserted MoE layers.             |
    | `moe-ffn-hidden-size`                 | Number of dimensions in the hidden FFN layer in MoE.           | `moe_intermediate_size`               | Number of dimensions in the hidden FFN layer in MoE.           |
    | `moe-shared-expert-intermediate-size` | Number of middle dimensions shared by experts.               | `moe_shared_expert_intermediate_size` | Number of middle dimensions shared by experts.               |
    | `moe-shared-expert-overlap`           | Specifies whether to overlap the middle layer shared by experts.               | `moe_shared_expert_overlap`           | Specifies whether to overlap the middle layer shared by experts.               |
    | `moe-grouped-gemm`                    | Specifies whether to use the grouped GEMM optimization.      | `use_gmm`                             | Specifies whether to use the grouped GEMM optimization.      |
    | `moe-router-load-balancing-type`      | Router load balancing policy.             | `moe_router_load_balancing_type`      | Router load balancing policy.             |
    | `moe-router-dtype`                    | Router score data type.             | `router_dense_type`                   | Router score data type.             |
    | `moe-router-score-function`           | Router score calculation method (for example, **softmax**).  | `use_gating_sigmoid`                  | Specifies whether to use the Sigmoid activation function.         |
    | `moe-router-topk`                     | Number of top-*k* selected routers.         | `num_experts_chosen`                  | Number of top-*k* selected routers.         |
    | `moe-router-pre-softmax`              | Specifies whether to preprocess before softmax.         | `moe_router_pre_softmax`              | Specifies whether to preprocess before softmax.         |
    | `moe-router-num-groups`               | Number of token groups.                 | `n_groups`                            | Number of token groups.                 |
    | `moe-router-group-topk`               | Number of top-*k* tokens in each group.       | `topk_group`                          | Number of top-*k* tokens in each group.       |
    | `moe-router-topk-scaling-factor`      | Top-*k* score scaling factor.              | `routed_scaling_factor`               | Top-*k* score scaling factor.              |
    | `moe-router-enable-expert-bias`       | Specifies whether to use the bias of an expert.        | `balance_via_topk_bias`               | Specifies whether to use the bias of an expert.        |
    | `moe-router-bias-update-rate`         | Update rate of expert bias.           | `topk_bias_update_rate`               | Update rate of expert bias.           |
    | `moe-use-legacy-grouped-gemm`         | Specifies whether to use the source version of Grouped GEMM.       | Not supported.                        |                            |
    | `moe-aux-loss-coeff`                  | Auxiliary loss coefficient of MoE.                | Not supported.                        |                            |
    | `moe-z-loss-coeff`                    | MoE z-loss coefficient.             | Not supported.                        |                            |
    | `moe-input-jitter-eps`                | Input jitter noise of MoE.         | `moe_input_jitter_eps`                | Input jitter noise of MoE.         |
    | `moe-token-dispatcher-type`           | Token scheduling policy (for example, **allgather**).   | `moe_token_dispatcher_type`           |  Token scheduling policy (for example, **allgather**).                          |
    | `moe-enable-deepep`                   | Specifies whether to enable DeepEP hybrid expert optimization.        | `moe_enable_deepep`                   | Specifies whether to enable DeepEP hybrid expert optimization.        |
    | `moe-per-layer-logging`               | Prints logs at each MoE layer.               | `moe_per_layer_logging`               | Prints logs at each MoE layer.               |
    | `moe-expert-capacity-factor`          | Expansion ratio of the expert capacity.             | `capacity_factor`                     | Expansion ratio of the expert capacity.             |
    | `moe-pad-expert-input-to-capacity`    | Specifies whether to fill the expert input to the capacity upper limit.       | `moe_pad_expert_input_to_capacity`    | Specifies whether to fill the expert input to the capacity upper limit.       |
    | `moe-token-drop-policy`               | Token discarding policy (for example, **probs** or **position**).| `enable_sdrop`                        | Token discarding policy (for example, **probs** or **position**).|
    | `moe-extended-tp`                     | Enables extended tensor parallelism.          | Not supported.                        |                            |
    | `moe-use-upcycling`                   | Specifies whether to enable expert upcycling.          | Not supported.                        |                            |
    | `moe-permute-fusion`                  | Enables internal permute fusion optimization of experts.       | `moe_permute_fusion`                  | Enables internal permute fusion optimization of experts.       |
    | `mtp-num-layers`                      | Number of MoE layers.                  | `mtp_depth`                           | Number of MoE layers.                  |
    | `mtp-loss-scaling-factor`             | Loss scaling in the MoE architecture.              | `mtp_loss_factor`                     | Loss scaling in the MoE architecture.              |

- Data loading and tokenization

    | Megatron-LM                   | Description                       | MindSpore Transformers | Description                            |
    |-------------------------------|---------------------------|------------------------|--------------------------------|
    | `data-path` and `split`        | General data path.                   | `data_path`            | Sampling ratio and path of the Megatron dataset.           |
    | `train-data-path`             | Training data path.                   | Not supported.                 |                                |
    | `valid-data-path`             | Validation data path.                   | Not supported.                 |                                |
    | `test-data-path`              | Test data path.                   | Not supported.                 |                                |
    | `vocab-size`                  | Vocabulary size.                     | `vocab_size`           | Vocabulary size.                          |
    | `vocab-file`                  | Vocabulary file path.                   | Not supported.                 |                                |
    | `merge-file`                  | BPE combination rule file.               | Not supported.                 |                                |
    | `tokenizer-type`              | Tokenizer type (for example, **GPT2BPETokenizer**).| Not supported.                 | The tokenizer corresponding to Hugging Face is used by default.|
    | `seq-length`                  | Input sequence length.                   | `seq_length`           | Input sequence length.                        |
    | `encoder-seq-length`          | Encoder input length.                  | Not supported.                 |                                |
    | `decoder-seq-length`          | Decoder input length.                  | Not supported.                 |                                |
    | `retriever-seq-length`        | Retriever sequence length (if enabled).            | Not supported.                 |                                |
    | `num-workers`                 | Number of threads for loading data.                 | `num_parallel_workers` | Number of threads for loading data.                      |
    | `num-dataset-builder-threads` | Number of threads for building datasets.                | Not supported.                 |                                |
    | `data-cache-path`             | Data cache path.                   | Not supported.                 |                                |

- Training control and save

    | Megatron-LM                    | Description                    | MindSpore Transformers                 | Description                                                                         |
    |--------------------------------|------------------------|----------------------------------------|-----------------------------------------------------------------------------|
    | Not supported.                         | Total number of local samples processed in each iteration.         | `batch_size`                           | Total number of local samples processed in each iteration, which is configured in `runner_wrapper`.                                         |
    | Not supported.                         | Total number of local samples processed in each iteration.         | `micro_batch_interleave_num`           | Number of micro-batch interleaving. When `micro_batch_interleave_num` is greater than 1, multiple copies are enabled for parallel processing.                          |
    | `global_batch_size`            | Total number of global samples processed in each iteration.         | `batch_size` and `data_parallel`        | Total number of global samples processed in each iteration, which is the value of `batch_size` multiplied by the value of `data_parallel` multiplied by the value of `micro_batch_interleave_num`.|
    | Not supported.                         | Number of iteration periods.                 | `epochs`                               | Number of iteration periods, which is configured in `runner_wrapper`.                                                 |
    | `train-samples`                | Total number of training samples.                | `sizes`                                | Total number of training samples, which is configured in `train_dataset`.                                                 |
    | `train-iters`                  | Total number of training iterations.               | `epochs`, `sizes`, and `global_batch_size`| Total number of training iterations, which is the value of `sizes` divided by the value of `global_batch_size` and multiplied by the value of `epochs`.                            |
    | `log-interval`                 | Log recording interval (number of iteration steps).          | `per_print_times`                      | Log recording interval (number of iteration steps), which is configured in `MFLossMonitor` of `callbacks`.                               |
    | `eval-iters`                   | Number of iterations used in each evaluation.          | Not supported.                                 |                                                                             |
    | `eval-interval`                | Number of evaluation interval steps.                | Not supported.                                 |                                                                             |
    | `save`                         | Model save path.                | `output_dir`                           | Model save path.                                                                     |
    | `save-interval`                | Model save interval (number of iteration steps).          | `save_checkpoint_steps`                | Model save interval (number of iteration steps), which is configured in `CheckpointMonitor` of `callbacks`.                           |
    | `non-persistent-save-interval` | (Non-persistent) temporary storage interval.          | Not supported.                                 |                                                                             |
    | `non-persistent-ckpt-type`     | Temporary storage type (for example, **global** or **local**).| Not supported.                                 |                                                                             |
    | `pretrained-checkpoint`        | Pretrained model path.               | Not supported.                                 |                                                                             |
    | `ckpt-step`                    | Loads the weight of a specified step.         | `load_checkpoint` and `resume_training` | Loads the weight of a specified name in resumable training scenarios.                                                          |
    | `load`                         | Loads a model from the path.              | `load_checkpoint`                      | Loads a model from the path.                                                                   |
    | `exit-interval`                | Iteration interval for exiting training.           | `stop_step`                            | Number of iterations after which the training is stopped, which is configured in `TrainCallMonitor` of `callbacks`.                              |
    | `exit-duration-in-mins`        | Interval for exiting training (in minutes).       | Not supported.                                 |                                                                             |

- Recomputation configurations

    The recomputation configuration logic of MindSpore Transformers is greatly different from that of Megatron-LM. For details, see [Recomputation](https://www.mindspore.cn/mindformers/docs/en/master/feature/memory_optimization.html#recomputation).

    | Megatron-LM                    | Description                   | MindSpore Transformers | Description                      |
    |--------------------------------|-----------------------|------------------------|--------------------------|
    | `recompute-activations`        | Specifies whether to enable activation recomputation to save memory.       | `recompute`            | Specifies whether to enable complete activation recomputation to save memory (`bool`).|
    | `recompute-granularity`        | Recomputation granularity (for example, **full** or **selective**).| `select_recompute`     | Specifies whether to enable selective recomputation.      |
    | `recompute-method`             | Recomputation method (for example, **uniform** or **block**). | Not supported.                 |                          |
    | `recompute-num-layers`         | Number of recomputation layers.               | `recompute`            | Number of recomputation layers (for example, `tuple` or `list`).  |
    | `distribute-saved-activations` | Distributed storage activation value.             | Not supported.                 |                          |
    | `checkpoint-activations`       | Specifies whether to enable the activation checkpoint mechanism to reduce the video RAM.    | Not supported.                 |                          |
    | `moe-layer-recompute`          | Enables recomputation at the MoE layer.           | Not supported.                 |                          |

**Note:** The two frameworks have other configurations that are not closely related to training. For details about MindSpore Transformers, see [Configuration Description](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html). You can run the `torchrun --nproc_per_node=1 pretrain_gpt.py --help` command to view the Megatron-LM configuration.

### 3.2 Dataset Alignment

In the precision comparison process, ensure that the two frameworks use the same data input. This section describes how to align the dataset creation and configuration of Megatron-LM and MindSpore Transformers to ensure the consistency of input samples, providing a basis for subsequent weight loading and precision validation.

#### 3.2.1 Preparing a Dataset

Both frameworks support loading the Megatron dataset. The dataset is preprocessed, serialized into a binary format (for example, `.bin` or `.idx` file), and accompanied by a specific indexing mechanism, which facilitates efficient parallel loading and data segmentation in a distributed cluster environment.

- Dataset download: [wikitext-103 dataset](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens)

- Tokenizer model download: [tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/tokenizer.json?download=true)

#### 3.2.2 Processing a Dataset

- Generating Megatron BIN files

   Place the dataset file `wiki.train.tokens` and the tokenization model file `tokenizer.json` in the `../dataset` directory, and create the `data.json` file by referring to [Megatron Dataset > Data Preprocessing](https://www.mindspore.cn/mindformers/docs/en/master/feature/dataset.html#data-preprocessing).

   Run the following commands to convert the dataset file into a BIN file:

   ```shell
   cd $MINDFORMERS_HOME
   python mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py \
    --input /path/data.json \
    --output-prefix ../dataset/wiki_4096 \
    --vocab-file ../dataset/tokenizer.json \
    --seq-length 4096 \
    --workers 1
   ```

- Building the Megatron BIN dataset module

   Run the following commands to build the Megatron BIN dataset module.

   ```shell
   pip install pybind11
   cd $MINDFORMERS_HOME/mindformers/dataset/blended_datasets
   make
   ```

   `$MINDFORMERS_HOME` indicates the directory where the MindSpore Transformers source code is stored.

#### 3.2.3 Configuring a Dataset

This section compares and describes the dataset configuration items in the configuration files of the two frameworks.

- Megatron-LM:

    The dataset configuration items in the Megatron-LM sample are as follows:

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

    In the preceding information:

    - `tokenizer-type`: type of the tokenization model file.
    - `tokenizer-model`: location of the tokenization model file `tokenizer.json`, which is accurate to the full file name.
    - `data-path`: location of the processed dataset, which is accurate to the prefix of the `.bin` or `.idx` file.
    - `split`: sampling ratio of the dataset.

- MindSpore Transformers:

    The dataset configuration items in the MindSpore Transformers sample are as follows:

    ```yaml
    config:  # GPTDataset configuration items.
      data_path:  # Sampling ratio and path of the Megatron dataset.
        - '1'
        - "/home/to/wiki_text_document"
    ```

    Note that the first parameter of `data_path` is the dataset sampling ratio, and the setting in the example is equivalent to `--split` in the Megatron-LM example. The second parameter is the location of the processed dataset, which is accurate to the prefix of the `.bin` or `.idx` file. The setting in the example is equivalent to `--data-path` in the Megatron-LM example.

### 3.3 Weight Alignment

To ensure the consistency of model behavior between different frameworks, the weights obtained after training must be accurately mapped to the corresponding positions in MindSpore Transformers and Megatron-LM through proper weight conversion and segmentation.

#### Weight Conversion

The weight formats, parameter naming modes, and tensor arrangements of MindSpore Transformers and Megatron-LM are different. Directly loading the weights will result in incompatibility. Therefore, you need to use a dedicated conversion script to convert the model weights exported from the source framework to the format that can be identified by the target framework.

1. Generating initial weights of MindSpore Transformers

   Modify the `example.yaml` file by referring to [Callbacks Configuration](https://www.mindspore.cn/mindformers/docs/en/master/feature/configuration.html#callbacks-configuration) and run the command provided in [Viewing Results](#3-4-viewing-results) to obtain an initial weight in `checkpoints` of `output_dir` in `example.yaml` through pre-training. The modification is as follows:

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

   **Note**: After obtaining the weight, restore `example.yaml`.

2. MindSpore Transformers to Megatron-LM

   To accurately map the weights of MindSpore Transformers to the equivalent weights that can be loaded by Megatron-LM, a weight conversion script is provided. You can obtain the equivalent weights by executing the weight conversion script.

### 3.4 Viewing Results

After the preceding steps are complete, you can start training and extract key data from the output result in the log to check the precision comparison result.

- Megatron-LM

  Save the `example.sh` file to the Megatron-LM code directory and run the following command:

  ```shell
  bash example.sh
  ```

- MindSpore Transformers

  Run the following commands in the MindSpore Transformers code directory:

  ```shell
  bash scripts/msrun_launcher.sh "run_mindformer.py \
   --config /path/to/example.yaml"
  ```

  `config` is the model configuration file, which is stored in the **config** directory of the MindSpore Transformers code repository.

- Result comparison

  View the output logs of the two models. The log path of Megatron-LM is `logs/${logtime}.log` in `example.sh`, and that of MindSpore Transformers is `msrun_log/worker_0.log` in `output_dir` of `example.yaml`. The following table lists the comparison results.

  | Megatron-LM     | MindSpore Transformers | Description                                                                                                                                                            |
  |-----------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | `iteration`     | `epoch` and `step`  | Number of global iterations during training. In MindSpore Transformers, `(epoch, step)` indicates the current training location, while Megatron-LM uses a single `iteration`. The relationship between them is as follows: `iteration = (epoch – 1) x steps_per_epoch + step`|
  | `lm loss`       | `loss`                 | Training loss, which is a core indicator in precision comparison. The value of `loss` of MindSpore Transformers is the sum of `lm loss` and `aux loss`. The values will be printed separately in the future.                                                                               |
  | `learning rate` | `lr`                   | Learning rate, which is the precision comparison reference indicator.                                                                                                                                                  |
  | `grad norm`    | `global norm`          | Global gradient norm, which is the precision comparison reference indicator.                                                                                                                                               |
