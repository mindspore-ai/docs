# 低参微调

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/usage/parameter_efficient_fine_tune.md)

## 概述

在深度学习模型的微调过程中，通常需要更新模型所有的权重，这会导致大量的计算资源消耗。LoRA（Low-Rank Adaptation）是一种通过将模型的部分权重矩阵分解为低秩矩阵来显著减少微调所需参数量的技术。结合华为昇腾AI处理器及MindSpore深度学习框架，以及MindFormers大模型套件，LoRA能够轻松应用于大规模预训练模型（如Llama2）的低参微调，提供高效且灵活的模型定制化能力。

## LoRA 原理简介

LoRA通过将原始模型的权重矩阵分解为两个低秩矩阵来实现参数量的显著减少。例如，假设一个权重矩阵W的大小为m x n，通过LoRA，该矩阵被分解为两个低秩矩阵A和B，其中A的大小为m x r，B的大小为r x n（r远小于m和n）。在微调过程中，仅对这两个低秩矩阵进行更新，而不改变原始模型的其他部分。
这种方法不仅大幅度降低了微调的计算开销，还保留了模型的原始性能，特别适用于数据量有限、计算资源受限的环境中进行模型优化，详细原理可以查看论文 [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) 。

## LoRA 微调的通用流程

在进行LoRA低参微调时，通常需要以下几个关键步骤：

1. **加载预训练模型权重**：从预训练模型中加载基础权重（Base Weights）。这些权重通常代表了模型在大规模数据集上进行训练后得到的参数。

2. **数据集准备**：选择并准备用于微调的数据集。数据集需要与目标任务相关，并且格式需要与模型的输入格式匹配。

3. **配置微调参数**：设置微调相关的参数，包括学习率、优化器类型、批次大小（batch_size）等。

4. **配置LoRA参数**：在模型的关键层（如注意力层）中配置 pet_config 参数，通过调整低秩矩阵来实现模型的参数更新。

5. **启动微调过程**：利用设置好的参数和数据集，在分布式环境中启动微调过程。

6. **评估与保存**：在微调过程中或结束后，评估模型的性能，并保存微调后的模型权重。

## 使用MindFormers进行Llama2的LoRA低参微调

在昇腾AI处理器的分布式环境中，可以通过MindFormers套件轻松实现LoRA的低参微调流程。以下展示了Llama2模型LoRA微调中的核心配置部分，并对 `pet_config` 参数进行了详细说明。

### 示例配置文件（YAML）

完整的YAML配置文件可以通过以下链接访问：[Llama2 LoRA微调 YAML 文件](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/lora_llama2_7b.yaml)。

```yaml
# model config
model:
  model_config:
    type: LlamaConfig
    batch_size: 1
    seq_length: 4096
    hidden_size: 4096
    num_layers: 32
    num_heads: 32
    vocab_size: 32000
    compute_dtype: "float16"
    pet_config:
      pet_type: lora
      lora_rank: 16
      lora_alpha: 16
      lora_dropout: 0.05
      target_modules: '.*wq|.*wk|.*wv|.*wo'
  arch:
    type: LlamaForCausalLM
```

### pet_config 参数详解

在 model_config 中，pet_config 是LoRA微调的核心配置部分，用于指定LoRA的相关参数。具体参数说明如下：

- **pet_type:** 指定参数高效微调技术（PET，Parameter-Efficient Tuning）的类型为LoRA。这意味着在模型的关键层中会插入LoRA模块，以减少微调时所需的参数量。
- **lora_rank:** 定义了低秩矩阵的秩值。秩值越小，微调时需要更新的参数越少，从而减少计算资源的占用。这里设为16是一个常见的平衡点，在保持模型性能的同时，显著减少了参数量。
- **lora_alpha:** 控制LoRA模块中权重更新的缩放比例。这个值决定了微调过程中，权重更新的幅度和影响程度。设为16表示缩放幅度适中，有助于稳定训练过程。
- **lora_dropout:** 设置LoRA模块中的dropout概率。Dropout是一种正则化技术，用于减少过拟合风险。设置为0.05表示在训练过程中有5%的概率会随机“关闭”某些神经元连接，这在数据量有限的情况下尤为重要。
- **target_modules:** 通过正则表达式指定LoRA将应用于模型中的哪些权重矩阵。在Llama中，这里的配置将LoRA应用于模型的自注意力机制中的Query（wq）、Key（wk）、Value（wv）和Output（wo）矩阵。这些矩阵在Transformer结构中扮演关键角色，插入LoRA后可以在减少参数量的同时保持模型性能。

通过配置这些参数，LoRA能够在进行微调时有效地减少计算资源的占用，同时保持模型的高性能表现。

### Llama2-7B 的 LoRA 微调示例

MindFormers 提供了 Llama2-7B 的 [LoRA 微调示例](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama2.md#lora%E5%BE%AE%E8%B0%83)。微调过程中使用的数据集可以参考[数据集下载](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)获得。

以 Llama2-7B 为例，可以执行以下 msrun 启动脚本，进行 8 卡分布式微调。

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/llama2_7b.ckpt \
 --auto_trans_ckpt False \
 --use_parallel True \
 --run_mode finetune" 8
```

如果加载的权重需要转换后才能加载，加载权重路径应设置为 rank_0 的上一层路径，同时开启权重自动转换功能 --auto_trans_ckpt True：

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --config configs/llama2/lora_llama2_7b.yaml \
 --train_dataset_dir /{path}/alpaca-fastchat4096.mindrecord \
 --load_checkpoint /{path}/checkpoint/ \
 --auto_trans_ckpt True \
 --use_parallel True \
 --run_mode finetune" 8
```


