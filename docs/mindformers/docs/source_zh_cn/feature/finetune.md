# 全参微调与 LoRA 微调

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/finetune.md)

MindSpore Transformers 动态图（PyNative）支持**全参微调**与 **LoRA 高效微调**。全参微调更新模型的全部参数；LoRA 微调冻结基座参数，仅训练注入到普通 `Linear` 和 MoE 路由专家 `GroupedMLP` 中的低秩适配器。

两种微调都通过 `run_mindformer.py --mode 1` 启动，与预训练共用 `PynativeTrainer`。开始前请准备：

- 与模型结构匹配的完整 Safetensors 基座权重；
- 模型的完整动态图 YAML，不能只使用本文展示的增量片段；
- 微调数据集和与基座模型匹配的 tokenizer；
- 已按[安装指南](../installation.md)安装 MindSpore Transformers 和 HyperParallel。

微调数据集的制作和配置请参考[数据集文档的 HuggingFace 数据集章节](./dataset.md#huggingface-数据集)。

## 一、全参微调

在MindSpore Transformers提供的 YAML 文件基础上修改 `checkpoint` 段：

```yaml
checkpoint:
  enable_save: true
  save_path: "./output/full_finetune"
  save_interleaved_steps: 500
  load_path: "/path/to/base_model"
  no_load_optim: true
```

| 字段 | 说明 |
|---|---|
| `load_path` | 加载完整基座权重。可以填写 HuggingFace 模型目录、训练保存根目录或 `iteration_*` 目录 |
| `no_load_optim` | 初次微调设为 `true`，只加载模型权重并新建优化器；断点续训时设为 `false` |

其余 `training`、`optimizer`、`lr_scheduler`、`model`、`parallelism` 和数据配置沿用目标模型的动态图训练 YAML。启动命令见[启动训练](#四启动训练)。

## 二、LoRA 微调

LoRA 在冻结的基座权重旁注入低秩矩阵。普通线性层的前向计算为：

$$y = x W^{T} + \frac{\alpha}{r}(x A^{T})B^{T}$$

其中 $W$ 冻结，只有 $A/B$ 参与训练。默认初始化为 $A \sim \mathcal{N}(0,\sigma^2)$、$B=0$，因此训练开始时 $\Delta W=0$，模型输出与基座模型一致。

### 2.1 配置基座加载和适配器保存

推荐只保存 LoRA 适配器，以减少 checkpoint 体积：

```yaml
checkpoint:
  enable_save: true
  save_path: "./output/lora_finetune"
  save_interleaved_steps: 500

  # 初次 LoRA 微调：从该目录加载完整基座权重
  base_load_path: "/path/to/base_model"
  load_path: ""

  # 只保存 LoRA 参数；加载时必须设置 base_load_path
  save_trainable_only: true

  # 保留优化器状态，便于断点续训
  no_save_optim: false
  no_load_optim: false
```

`save_trainable_only: true` 时，仅保存适配器参数，同时保存 `mindformers_adapter_config.json` 文件，记录 LoRA 配置、基座路径和基座指纹。加载或续训时，需要同时配置 `base_load_path` 和 `load_path`。

`save_trainable_only: false`时，保存完整基座参数和适配器参数，checkpoint 体积与全参保存接近。

### 2.2 配置 LoRA 注入目标

在 YAML 顶层增加 `lora_config`：

```yaml
lora_config:
  target_modules: ".*linear_qkv.*|.*linear_proj.*|.*linear_fc1.*|.*linear_fc2.*"
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.0
```

| 字段 | 默认值 | 说明 |
|---|---|---|
| `target_modules` | `""` | 必填正则表达式。点分模块路径匹配该正则的 `Linear` 或 `GroupedMLP` 会被替换为 LoRA 层；未命中任何支持层时直接报错 |
| `exclude_layers` | `None` | 可选正则表达式；排除已被 `target_modules` 命中的模块 |
| `lora_rank` | `8` | 适配器秩 $r$ |
| `lora_alpha` | `16` | 缩放分子，有效缩放为 $\alpha/r$ |
| `lora_dropout` | `0.0` | 适配器分支输入的 dropout 概率 |
| `lora_a_std` | `0.01` | `lora_a` 的正态初始化标准差 |
| `lora_b_std` | `0.0` | `lora_b` 的正态初始化标准差；常规微调保持 `0.0` |

普通动态图 Transformer 常见的目标名包括 `linear_qkv`、`linear_proj`、`linear_fc1` 和 `linear_fc2`。仅注入注意力层可配置为：

```yaml
lora_config:
  target_modules: ".*linear_qkv.*|.*linear_proj.*"
```

MoE 路由专家由 `GroupedMLP` 实现，模块名通常以 `experts` 结尾。只为路由专家注入 LoRA 可配置为：

```yaml
lora_config:
  target_modules: '.*\.experts$'
```

同时匹配普通线性层和路由专家时，用 `|` 合并正则。例如，为注意力层和 MoE 路由专家同时注入 LoRA 可配置为：

```yaml
lora_config:
  target_modules: '.*linear_qkv.*|.*linear_proj.*|.*\.experts$'
```

> 不同模型的模块命名可能不同，应以目标模型代码和启动日志中的模块路径为准。

### 2.3 LoRA 断点续训

使用仅适配器 checkpoint 续训时，同时填写原始基座目录和适配器保存目录：

```yaml
checkpoint:
  base_load_path: "/path/to/base_model"
  load_path: "./output/lora_finetune"
  save_path: "./output/lora_finetune"
  save_trainable_only: true
  no_save_optim: false
  no_load_optim: false
```

框架先校验 `mindformers_adapter_config.json` 中记录的基座指纹，再依次加载基座、适配器和优化器状态。

使用全量 checkpoint 续训时，`load_path` 指向全量保存目录，不需要配置 `base_load_path`：

```yaml
checkpoint:
  load_path: "./output/lora_finetune"
  save_path: "./output/lora_finetune"
  save_trainable_only: false
  no_save_optim: false
  no_load_optim: false
```

框架从 `load_path` 恢复基座参数、LoRA 参数、优化器状态和训练步数，续训时的 `lora_config` 需要与保存 checkpoint 时保持一致。

## 三、合并 LoRA 权重

LoRA 权重合并用于将适配器增量合入基座参数，生成不依赖 LoRA 适配器的完整权重，便于推理或部署。

仅适配器保存模式需要分别传入基座和适配器目录：

```bash
python mindformers/tools/transform_safetensors_lora.py \
  --base_ckpt_path /path/to/base_model \
  --adapter_ckpt_path ./output/lora_finetune \
  --dst_ckpt_path ./output/merged.safetensors
```

| 参数 | 说明 |
|---|---|
| `--base_ckpt_path` | 完整基座权重路径。支持单个 Safetensors 文件、单卡目录、带 tracker 的训练保存根目录，以及含 `metadata.json` 的分布式目录 |
| `--adapter_ckpt_path` | 可选的独立 LoRA checkpoint。仅适配器保存模式必须填写；工具会校验其基座指纹 |
| `--dst_ckpt_path` | 合并后的输出文件；`.safetensors` 后缀可省略，输出始终是单个完整权重文件 |
| `--lora_alpha` | 可选。默认从 `mindformers_adapter_config.json` 或 `adapter_config.json` 读取；无法读取时使用 `16`。显式传入时必须与训练配置一致 |

如果 checkpoint 同时包含基座和 LoRA 参数，可省略 `--adapter_ckpt_path`：

```bash
python mindformers/tools/transform_safetensors_lora.py \
  --src_ckpt_path ./output/lora_full_checkpoint \
  --dst_ckpt_path ./output/merged.safetensors \
  --lora_alpha 16
```

`--src_ckpt_path` 是 `--base_ckpt_path` 的兼容别名。合并工具支持普通线性层 LoRA 和 MoE `GroupedMLP` 专家 LoRA，并自动忽略优化器分片。

## 四、启动训练

在 MindSpore Transformers 仓库根目录执行：

```bash
# 单卡
python run_mindformer.py --config /path/to/finetune.yaml --mode 1

# 单机 8 卡
bash scripts/msrun_launcher.sh \
  "run_mindformer.py --config /path/to/finetune.yaml --mode 1" 8
```

多卡时，`dp_replicate * dp_shard * cp * tp * pp` 必须等于进程数；`global_batch_size`、`local_batch_size` 与并行维度也必须满足[分布式并行训练](./parallel_training.md)中的约束。启动日志出现 `Running MindFormers in PYNATIVE_MODE.` 表示已进入动态图训练。

## 五、全参微调与 LoRA 微调对比

| 维度 | 全参微调 | LoRA 微调 |
|---|---|---|
| 可训练参数 | 模型全部参数 | 仅 LoRA 适配器，基座冻结 |
| 优化器状态和显存 | 高 | 较低 |
| 基座加载 | `checkpoint.load_path` | 推荐使用 `checkpoint.base_load_path` |
| 保存产物 | 完整模型 checkpoint | 推荐只保存适配器及其清单 |
| 部署前处理 | 可直接使用匹配配置的完整权重 | 仅适配器模式需先与基座合并 |

## 相关文档

- [启动任务](./start_task.md)
- [配置文件说明](./configuration.md)
- [权重保存与加载](./save_load_checkpoint.md)
- [断点续训](./resume_training.md)
- [数据集](./dataset.md)
- [分布式并行训练](./parallel_training.md)
- [训练超参数与优化器](./training_hyperparameters.md)
