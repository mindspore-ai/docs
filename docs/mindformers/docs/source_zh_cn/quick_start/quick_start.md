# 快速开始

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/quick_start/quick_start.md)

## 概述

本章帮助你快速使用动态图（PyNative）完成一次大模型训练，建立对整体流程的直观认识。我们以 **DeepSeek-V3** 为例，在 2 卡上执行最小规模的预训练。

动态图训练以统一脚本 `run_mindformer.py` 为入口，通过 `--mode 1` 路由到动态图训练器 `mindformers.pynative.trainer.Trainer`，由其完成模型、数据集、优化器的构建并驱动训练循环。一次任务可归纳为三步：

1. **准备配置文件** —— 编写一份动态图 YAML，把模型、数据、并行、优化器串起来；
2. **启动训练** —— 使用 `msrun` 完成多卡启动；
3. **查看结果** —— 通过日志确认每步 loss/grad norm 正常下降。

完整流程与各能力的详细配置见 [功能特性](../feature/overview.md)（训练指南与各特性页正文将随后续提交上线）。

## 前置条件

- 已安装 MindSpore 与 MindSpore Transformers，详见 [安装指南](../installation.md)；
- 昇腾（Ascend）硬件环境，且已正确配置 CANN；
- 已准备 **Megatron 格式数据集**（`.bin`/`.idx`）。数据集制作（json → bin/idx）可用仓库脚本 [`preprocess_indexed_dataset.py`](https://atomgit.com/mindspore/mindformers/blob/master/toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py)，详见「数据集」文档（将随后续提交上线）。

> **数据路径如何衔接配置**
>
> 预处理产出的是一对 `xxx.bin` 与 `xxx.idx`，文件名形如 `<output-prefix>_text_document.bin`（脚本默认追加 `_text_document`）。本页 YAML 中 `train_dataset.dataloader.config.data_path` 要填**含该后缀的前缀**。例如 `--output-prefix /path/megatron_data` 会生成 `/path/megatron_data_text_document.bin`，配置里就填 `/path/megatron_data_text_document`。同时记下预处理时使用的 tokenizer 对应的 `eod`（eos）token id，下面 `config.eod` 要用到。

## 第一步：准备配置文件

动态图使用 dataclass 风格的 YAML 配置，顶层各段分别对应权重、训练、并行、优化器、学习率、数据、模型等。本页配套提供完整的 2 卡示例配置 [`pynative_ds3.yaml`](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/quick_start/pynative_ds3.yaml)，可直接下载使用；其各段内容如下（完整字段说明见「配置文件说明」文档）：

```yaml
checkpoint:
  enable_save: False              # 快速验证可先不保存权重
  save_path: "./output/ds3"

training:
  steps: 10                       # 训练步数
  local_batch_size: 1             # 每卡（每次前向）的 batch size
  global_batch_size: 2            # 全局 batch size，见下方说明
  max_norm: 1.0                   # 梯度裁剪（> 0 时生效）
  seed: 42

parallelism:
  data_parallel_shard: -1         # FSDP，-1 表示按可用卡数自动切分
  expert_parallel: 1
  tensor_parallel: 1
  context_parallel: 1
  pipeline_parallel: 1
  # SP 随 TP 自动开启，当前不支持关闭（无需配置此项，详见分布式并行训练文档）

optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
  weight_decay: 0.01

lr_scheduler:
  type: ConstantWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0

train_dataset:
  dataloader:
    type: BlendedMegatronDatasetDataLoader
    datasets_type: "GPTDataset"
    sizes: [1000, 0, 0]           # [训练, 测试, 评估] 样本数；当前仅训练集生效
    column_names: ["input_ids", "labels", "loss_mask", "position_ids"]
    shuffle: false
    config:
      seed: 1234
      seq_length: 4096            # 须与 model.seq_length 一致
      split: "1, 0, 0"            # [训练,测试,评估] 划分；设了 data_path 即必填，缺失会报错
      eod: 1                      # 数据集中 eod(eos) 的 token id
      pad: -1                     # 数据集中 pad 的 token id
      eod_mask_loss: False
      reset_position_ids: False
      create_attention_mask: False  # 输出 4 列；若设 True 需在 column_names 增列 attention_mask
      reset_attention_mask: False
      create_compressed_eod_mask: False
      eod_pad_length: 128
      data_path:                  # 采样权重(相对值,自动归一化) + bin 前缀(含 _text_document)
        - '1'
        - "/path/megatron_data_text_document"
  drop_remainder: True
  num_parallel_workers: 8

model:
  model_type: deepseek_v3
  architectures: DeepseekV3ForCausalLM
  seq_length: 4096                # 须与 train_dataset.dataloader.config.seq_length 一致
  # ↓ 其余模型结构超参（hidden_size/num_hidden_layers/MoE 等）见上方链接的完整示例配置（平铺写法）

```

> `global_batch_size` 的精确含义：框架按 `num_accumulation_steps = global_batch_size / (data_parallel_shard × local_batch_size)` 推导梯度累积步数。未启用梯度累积时（即三者相乘恰好相等），`global_batch_size = local_batch_size × 数据并行数`；一旦 `global_batch_size` 大于该乘积，多出的倍数即为梯度累积步数。精确定义见「配置文件说明」文档。

### 关于数据集段

`BlendedMegatronDatasetDataLoader` 运行时**必须**有 `datasets_type`、`sizes` 以及嵌套的 `config` 块（含 `seq_length`/`split`/`eod`/`pad`/`data_path`/`create_compressed_eod_mask` 等），缺一不可。其中：

| 字段 | 说明 |
|---|---|
| `datasets_type` | 数据集类型，预训练用 `"GPTDataset"` |
| `sizes` | `[训练, 测试, 评估]` 样本数，当前仅训练集生效 |
| `config.seq_length` | 返回序列长度，**须与 `model.seq_length` 一致** |
| `config.split` | 训练/测试/评估划分比例（如 `"1, 0, 0"`）；设了 `data_path` 即必填 |
| `config.eod` / `config.pad` | eod(eos)/pad 的 token id，取自预处理时的 tokenizer |
| `config.data_path` | 列表，每两个元素为一组「采样权重, bin 前缀」，权重为相对值、自动归一化（不要求之和为 1）；bin 前缀含 `_text_document` 后缀 |

> 这些字段的逐项含义、多数据源混合、压缩 EOD mask 等场景化配置见「数据集」文档。本页只给最小可跑配置。

### 关于模型段

DeepSeek-V3 的 `model` 段包含数十个结构超参（`hidden_size`、`num_hidden_layers`、MoE 路由等），逐一手写既冗长又易错。本页配套的完整示例配置 [`pynative_ds3.yaml`](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/quick_start/pynative_ds3.yaml) 已包含全部 `model` 段字段（dataclass 风格、结构超参平铺），直接下载使用即可；仅需确认 `seq_length` 与数据集一致。

> 注意：`configs/deepseek3/` 下的配置是静态图 legacy 结构（结构超参嵌套在 `model.model_config` 下、`architectures` 为列表、`context.mode: 0`），不能整段复制到动态图配置——需把结构超参「上提一层」平铺到 `model:` 下，并把 `architectures` 改成字符串。

## 第二步：启动训练

动态图训练通过 `msrun` 启动多卡。以下命令在 **mindformers 仓库根目录**下执行（将下载的 `pynative_ds3.yaml` 放在该目录），在 2 卡上启动训练：

```bash
msrun --worker_num=2 --local_worker_num=2 --master_port=8118 \
      --join=True --log_dir=./msrun_log \
      run_mindformer.py --config pynative_ds3.yaml --mode 1
```

- `--worker_num` / `--local_worker_num`：总卡数 / 本机卡数；
- `--config`：上一步的 YAML；
- `--mode 1`：使用动态图。

> 进入动态图训练**必须在启动命令显式传入 `--mode 1`**（入口只读取命令行 `--mode`，不读取 YAML）。YAML 的 `context` 段可省略：默认即动态图模式（`mode: 1`）、Ascend 后端；如需调整 `max_device_memory` 等再显式配置。

更多集群规模（单卡、多机）的启动方式见「启动任务」文档。

## 第三步：查看结果

训练日志输出在 `--log_dir` 指定目录下，每个 worker 一份子目录（如 `./msrun_log/worker_0.log`）。打开任一 worker 日志，看到形如下面的逐步输出即说明训练已正常运行：

```text
{ step:[    1/   10], loss:  11.813965, per_step_time:  13570ms, load_balancing_loss:   1.093977, lr: 1.000000e-05, grad_norm:  13.831877, throughput:   1.36T }
{ step:[    2/   10], loss:  11.755612, per_step_time:    710ms, load_balancing_loss:   1.106543, lr: 1.000000e-05, grad_norm:  19.926786, throughput:  25.92T }
```

判断要点：

- `step` 持续递增、`loss` 整体呈下降趋势（前几步可能波动）；
- `grad_norm` 是有限数值而非 `NaN`；
- `per_step_time` 在首步后趋于稳定（首步含初始化开销，通常显著更慢）；
- MoE 模型（如本例 DeepSeek-V3）会额外打印 `load_balancing_loss`。

此外：

- **权重**：若开启 `checkpoint.enable_save`，权重以 Safetensors 格式保存到 `checkpoint.save_path`；
- **更多指标**：可通过 `monitor.train_state` 采集逐参数范数与本地/设备级 loss（注意：动态图监控指标当前仅通过训练日志输出，TensorBoard 配置暂不生效），详见「训练指标监控与 Profiling」文档。

## 相关文档

- 动态图整体架构：[整体架构](../introduction/overview.md)
- 当前支持的模型：[模型支持库](../introduction/models.md)
- 各能力一览：[功能特性概述](../feature/overview.md)

> 配置文件说明、数据集、分布式并行训练、训练指南等页面正文将随后续提交上线。
