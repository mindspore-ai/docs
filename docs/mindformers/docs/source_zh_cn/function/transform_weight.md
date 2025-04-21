# 分布式权重切分与合并

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/function/transform_weight.md)

## 概述

在当前的分布式训练和推理环境中，当预训练权重与分布式策略不匹配时，需要对预训练权重进行转换，以适应相应的分布式策略。为满足不同场景下的权重转换需求，MindSpore Transformers提供了一套权重转换工具。该工具支持单卡权重切分为多卡权重、多卡权重之间的转换、多卡权重合并为单卡权重。用户可根据具体需求选择[自动转换](#自动转换)或[离线转换](#离线转换)，帮助模型在不同分布式场景之间快速切换。

此外，MindSpore Transformers还支持[LoRA权重的合并](#lora权重合并)，方便用户部署使用LoRA微调后的模型。

## 自动转换

模型加载权重时，自动转换功能可以自动检测权重与当前模型分布式切分策略之间的匹配情况，如果不匹配，自动进行权重转换，无需用户手动干预。

### 参数说明

**自动权重转换**相关`yaml`文件参数说明如下：

| 参数名称              | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| load_checkpoint     | 预加载权重的绝对路径或文件夹路径。<br> - 如果是完整权重，则填写绝对路径；<br> - 如果是分布式权重，则填写文件夹路径，分布式权重须按照`model_dir/rank_x/xxx.ckpt`格式存放，文件夹路径填写为`model_dir`。<br>**如果rank_x文件夹下存在多个ckpt，将会使用文件名默认排序最后的ckpt文件用于转换。**                                                                                                                                                                                                                                                |
| src_strategy_path_or_dir        | 预加载权重对应的[分布式策略文件](#生成分布式策略)路径。<br> - 如果预加载权重是完整权重，则**不填写**；<br> - 如果预加载权重是分布式权重，且预加载权重保存时使用了流水线并行，则填写**合并的策略文件路径**或**分布式策略文件夹路径**；<br> - 如果预加载权重是分布式权重，且预加载权重保存时未使用流水线并行，则填写任一**ckpt_strategy_rank_x.ckpt**路径；                                                                                                                                                                                                                     |
| auto_trans_ckpt     | 权重自动转换开关，为True开启，默认False。                                                                                                                                                                                                                                                                                                                                                                                                          |
| transform_process_num | 权重自动转换使用的进程数，默认为1。<br> - 如果transform_process_num = 1，使用**单进程转换**，转换时只有rank_0负责权重转换，其他进程等待rank_0转换结束；<br> - 如果transform_process_num > 1，使用**多进程转换**，比如8卡任务，transform_process_num=2时，转换时rank_0负责rank_0/1/2/3切片权重的转换，rank_4负责rank_4/5/6/7切片权重的转换，其他进程等待rank_0/4转换结束；<br>**注意**：<br> 1. transform_process_num越大，转换时间越短，**转换所占用的host内存越大**；当出现host侧内存不足时，需要减少transform_process_num。<br> 2. transform_process_num必须能够整除NPU卡数，且最大不得超过NPU卡数。 |
| transform_by_rank   | 是否使用mindspore.transform_checkpoint_by_rank接口做权重转换。<br> - transform_process_num > 1时，自动设置为`True`；<br> - transform_process_num = 1时，如果目标权重为分布式权重，则循环调用mindspore.transform_checkpoint_by_rank串行转换每一个rank切片权重。<br>- transform_process_num = 1时，如果目标权重为完整权重，则自动设置为`False`，使用mindspore.transform_checkpoints接口做权重转换；                                                                                                                     |

### 不同场景下yaml配置说明

#### 单卡权重切分为多卡权重

```yaml
# load_checkpoint: 设置为预训练权重文件路径
load_checkpoint: "/worker/llama3_8b/llama3_8b.ckpt"

# auto_trans_ckpt: 开启自动转换
auto_trans_ckpt: True
```

#### 多卡权重之间的转换

```yaml
# load_checkpoint: 设置为多卡权重文件夹路径
load_checkpoint: "/worker/checkpoint/llama3-8b-2layer-dp2mp2pp2"

# src_strategy_path_or_dir: 设置为分布式策略文件路径
src_strategy_path_or_dir: "/worker/checkpoint/llama3-8b-2layer-dp2mp2pp2/strategy/merged_ckpt_strategy.ckpt"

# auto_trans_ckpt: 开启自动转换
auto_trans_ckpt: True
```

#### 多卡权重合并为单卡权重

```yaml
# load_checkpoint: 设置为多卡权重文件夹路径
load_checkpoint: "/worker/checkpoint/llama3-8b-2layer-dp1mp2pp2"

# src_strategy_path_or_dir: 设置为分布式策略文件路径
src_strategy_path_or_dir: "/worker/checkpoint/llama3-8b-2layer-dp1mp2pp2/strategy/merged_ckpt_strategy.ckpt"

# auto_trans_ckpt: 开启自动转换
auto_trans_ckpt: True

# use_parallel: 设置为False
use_parallel: False
```

#### 开启多进程转换（可选）

```yaml
# transform_process_num: 设置参与转换的进程数量
transform_process_num: 2
```

### 注意事项

- **多进程转换**：配置`transform_process_num`参数以开启多进程转换，但需注意内存占用。如果发生内存溢出，建议降低进程数量。

- **自动权重转换**：开启自动转换后，系统将删除`output`目录下的旧`strategy`和`transformed_checkpoint`文件夹，并保存当前任务的输出结果。建议在转换任务结束后，将`strategy`和`transformed_checkpoint`文件夹移动到自定义目录，以避免后续操作中被误删。

- **分布式策略文件保存**：分布式策略文件将保存在`output/strategy`文件夹下。如果开启了**流水线并行**，系统会自动合并所有的`ckpt_strategy_rank_x.ckpt`文件，生成`merged_ckpt_strategy.ckpt`。如果未开启流水线并行，则不会进行合并操作。

## 离线转换

离线转换功能旨在满足用户手动转换权重的需求。通过离线转换，用户可以在独立的环境中进行模型权重的转换操作。离线转换支持多种权重转换场景，包括单卡权重切分为多卡权重、多卡权重之间的转换、多卡权重合并为单卡权重。

用户在使用离线转换时，可以根据具体需求手动配置转换参数，确保转换过程灵活且可控，尤其适用于在严格控制的计算环境中进行模型部署和优化的场景。

### 参数说明

**离线权重转换**相关`yaml`参数说明如下：

| 参数名称        | 说明        |
| ----------------- |-----------------------------|
| src_checkpoint | 源权重的绝对路径或文件夹路径。<br> - 如果是**完整权重**，则填写**绝对路径**；<br> - 如果是**分布式权重**，则填写**文件夹路径**，分布式权重须按照`model_dir/rank_x/xxx.ckpt`格式存放，文件夹路径填写为`model_dir`。<br>**如果rank_x文件夹下存在多个ckpt，将会使用文件名默认排序最后的ckpt文件用于转换。** |
| src_strategy_path_or_dir   | 源权重对应的分布式策略文件路径。<br> - 如果是完整权重，则**不填写**；<br> - 如果是分布式权重，且使用了流水线并行，则填写**合并的策略文件路径**或**分布式策略文件夹路径**；<br> - 如果是分布式权重，且未使用流水线并行，则填写任一**ckpt_strategy_rank_x.ckpt**路径；                                 |
| dst_checkpoint | 保存目标权重的文件夹路径。           |
| dst_strategy   | 目标权重对应的分布式策略文件路径。<br> - 如果是完整权重，则**不填写**；<br> - 如果是分布式权重，且使用了流水线并行，则填写**合并的策略文件路径**或**分布式策略文件夹路径**；<br> - 如果是分布式权重，且未使用流水线并行，则填写任一**ckpt_strategy_rank_x.ckpt**路径；           |
| prefix          | 目标权重保存的前缀名，权重保存为”{prefix}rank_x.ckpt”，默认”checkpoint_”。 |
| world_size     | 目标权重的切片总数，一般等于dp \* mp \* pp。   |
| process_num    | 离线权重转换使用的进程数，默认为1。<br> - 如果process_num = 1，使用**单进程转换**；<br>- 如果process_num > 1，使用**多进程转换**，比如转换的目标权重为8卡分布式权重，process_num=2时，会启动两个进程分别负责rank_0/1/2/3和rank_4/5/6/7切片权重的转换；                          |

### 离线转换配置说明

#### 生成分布式策略

MindSpore每次运行分布式任务后都会在`output/strategy`文件夹下生成对应卡数的分布式策略文件（ckpt格式），可以在离线权重转换中使用。

如果当前没有分布式策略文件，可以通过这种方式快速生成：在原有分布式训练/推理任务的基础上，在yaml配置文件中设置`only_save_strategy:True`来生成策略文件。设置之后任务会在生成分布式策略文件后立即停止，而不会实际执行训练或推理。

#### 单进程转换

使用[mindformers/tools/ckpt_transform/transform_checkpoint.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/mindformers/tools/ckpt_transform/transform_checkpoint.py)对载入权重进行单进程转换。

**运行命令**：

```shell
python transform_checkpoint.py \
  --src_checkpoint /worker/checkpoint/llama3-8b-2layer/rank_0/llama3_8b.ckpt \
  --dst_checkpoint /worker/transform_ckpt/llama3_8b_1to8/ \
  --dst_strategy /worker/mindformers/output/strategy/
```

#### 多进程转换

使用[mindformers/tools/ckpt_transform/transform_checkpoint.sh](https://gitee.com/mindspore/mindformers/blob/r1.5.0/mindformers/tools/ckpt_transform/transform_checkpoint.sh)对载入权重进行多进程转换。

**运行命令**：

```shell
bash transform_checkpoint.sh \
  /worker/checkpoint/llam3-8b-2layer/rank_0/llama3_8b.ckpt \
  None \
  /worker/transform_ckpt/llama3_8b_1to8/ \
  /worker/mindformers/output/strategy/ \
  8 2
```

**注意事项**：

- 使用[transform_checkpoint.sh](https://gitee.com/mindspore/mindformers/blob/r1.5.0/mindformers/tools/ckpt_transform/transform_checkpoint.sh)脚本时，参数`8`表示目标设备数，参数`2`表示使用2个进程进行转换。

## 特殊场景

### 物理机多机多卡训练

大规模模型通常需要通过多台服务器组成的集群进行训练。在这种多机多卡的场景下，如果服务器之间存在共享盘，则可以使用自动转换功能，否则只能使用离线转换。下面以两台服务器、16卡训练为例进行说明。

#### 场景一：服务器之间有共享盘

在服务器之间有共享盘的场景下，可以使用 MindSpore Transformers 的自动权重转换功能在多机多卡训练之前自动进行权重转换。假设 `/data` 为服务器的共享盘，且 MindSpore Transformers 的工程代码位于 `/data/mindformers` 路径下。

- **单进程转换**

  在单进程转换模式下，只需在配置文件中配置预训练权重的路径并开启自动权重转换即可。

  **参数配置：**

  ```yaml
  # 配置预训练权重路径，填写权重文件的绝对路径
  load_checkpoint: "/worker/checkpoint/llama3-8b/rank_0/llama3_8b.ckpt"

  # 设置 auto_trans_ckpt 为 True 开启自动权重转换
  auto_trans_ckpt: True

  # 配置数据集路径
  train_dataset: &train_dataset
    data_loader:
      type: MindDataset
      dataset_dir: "/worker/dataset/wiki103/"
      shuffle: True

  # 配置16卡分布式策略（仅供参考）
  parallel_config:
    data_parallel: 2
    model_parallel: 4
    pipeline_stage: 2
    micro_batch_num: 2
    vocab_emb_dp: True
    gradient_aggregation_group: 4
    micro_batch_interleave_num: 1
  ```

- **多进程转换（可选）**

  若需要加速权重转换过程，可以选择多进程转换模式，通过配置 `transform_process_num` 参数实现。

  **参数配置：**

  ```yaml
  # 使用2个进程进行转换
  transform_process_num: 2
  ```

  **启动任务：**

  使用[mindformers/scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/r1.5.0/scripts/msrun_launcher.sh)进行任务启动。

  ```shell
  # 第一台服务器（主节点）
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode train" \
    16 8 ${ip} ${port} 0 output/msrun_log False 300
  # 第二台服务器（子节点）
  bash scripts/msrun_launcher.sh "run_mindformer.py \
    --config {CONFIG_PATH} \
    --run_mode train" \
    16 8 ${ip} ${port} 1 output/msrun_log False 300
  ```

#### 场景二：服务器之间无共享盘

在服务器之间无共享盘的情况下，需要使用离线权重转换工具进行权重转换。以下步骤描述了如何进行离线权重转换，并启动多机多卡训练任务。

- **获取分布式策略文件**

  在进行离线权重转换前，首先需要获取各节点的分布式策略文件。

  **参数配置：**

  ```yaml
  # 设置 only_save_strategy 为 True 以获取分布式策略文件
  only_save_strategy: True

  # 配置数据集路径
  train_dataset: &train_dataset
    data_loader:
      type: MindDataset
      dataset_dir: "/worker/dataset/wikitext_2048/"
      shuffle: True

  # 配置16卡分布式策略（仅供参考）
  parallel_config:
    data_parallel: 2
    model_parallel: 4
    pipeline_stage: 2
    micro_batch_num: 2
    vocab_emb_dp: True
    gradient_aggregation_group: 4
    micro_batch_interleave_num: 1
  ```

  各节点的策略文件将分别保存在各自的`output/strategy`目录中。例如，节点0将保存`ckpt_strategy_rank_0-7.ckpt`文件，节点1将保存`ckpt_strategy_rank_8-15.ckpt`文件。随后，需将所有节点的策略文件集中到同一台服务器上，以便进行后续操作。

- **离线权重转换**

  在保存有所有策略文件的服务器上，使用[mindformers/tools/ckpt_transform/transform_checkpoint.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/mindformers/tools/ckpt_transform/transform_checkpoint.py)进行离线权重转换。

  **单进程转换：**

  ```shell
  python mindformers/tools/ckpt_transform/transform_checkpoint.py \
    --src_checkpoint /worker/checkpoint/llama3-8b/rank_0/llama_7b.ckpt \
    --dst_checkpoint ./output/llama3_8b_dp2mp4pp2 \
    --dst_strategy ./output/strategy
  ```

  **多进程转换（可选）：**

  ```shell
  # 使用2个进程进行转换
  bash mindformers/tools/ckpt_transform/transform_checkpoint.sh \
    /worker/checkpoint/llama3-8b/rank_0/llama_7b.ckpt \
    None \
    ./output/llama3_8b_dp2mp4pp2 \
    ./output/strategy \
    16 2
  ```

- **复制权重到其他节点**

  将转换得到的分布式权重分别复制到各自节点。0节点只需要 `rank_0` 到 `rank_7` 的切片权重，1节点只需要 `rank_8` 到 `rank_15` 的切片权重。

- **参数配置**

  ```yaml
  # 配置预训练权重路径，填写分布式权重文件夹路径 model_dir
  load_checkpoint: "/worker/checkpoint/llama3_8b_dp2mp4pp2"

  # 将 only_save_strategy 改为 False
  only_save_strategy: False
  ```

### ModelArts 训练

在 ModelArts 环境中进行训练与物理机上的多机多卡训练类似，同样支持开启权重自动转换。用户可以通过在训练作业的超参数中配置`auto_trans_ckpt=True`来启用自动权重转换，并通过设置`transform_process_num > 1`来开启多进程转换。

**注意**：如果 ModelArts 资源池中的服务器节点NPU卡数不是8，则需要额外配置`npu_num_per_node=节点NPU卡数`。例如，如果每个节点配有16个NPU，则应设置`npu_num_per_node=16`。

## LoRA权重合并

### 概述

LoRA（Low-Rank Adaptation）的基本原理是对原始模型的参数进行低秩重参数化。合并LoRA权重的核心过程是将 LoRA 分支的参数进行计算，并叠加到对应的模型参数中，使最终得到的权重文件的参数列表与原始模型一致，不包含额外的LoRA参数。这一操作不会对推理结果产生任何影响，因此合并后的模型在推理时依然能够保持与原始模型一致的性能。
有关 LoRA 的详细原理和实现，请参阅以下资源：

- 论文: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- GitHub: [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)

### 使用说明

使用MindSpore Transformers提供的[LoRA权重合并脚本](https://gitee.com/mindspore/mindformers/blob/r1.5.0/mindformers/tools/transform_ckpt_lora.py)，按照如下方式进行LoRA权重合并。

```shell
python mindformers/tools/transform_ckpt_lora.py \
  --src_ckpt_strategy src_strategy_path_or_dir \
  --src_ckpt_path_or_dir src_ckpt_path_or_dir \
  --dst_ckpt_dir dst_ckpt_dir \
  --prefix "checkpoint_" \
  --lora_scaling lora_alpha/lora_rank
```

#### 参数说明

- **src_ckpt_strategy**：源权重对应的分布式策略文件路径，通常在启动训练任务后默认保存在 `output/strategy/` 目录下。如果源权重为完整权重，则无需填写此参数；如果为分布式权重，需根据以下情况填写：
    - **源权重开启了流水线并行**：权重转换基于合并的策略文件，填写分布式策略文件夹路径。脚本会自动将文件夹内的所有 `ckpt_strategy_rank_x.ckpt` 文件合并，并在文件夹下生成 `merged_ckpt_strategy.ckpt`。如果已经存在 `merged_ckpt_strategy.ckpt`，可以直接填写该文件的路径。
    - **源权重未开启流水线并行**：权重转换可基于任一策略文件，填写任意一个 `ckpt_strategy_rank_x.ckpt` 文件的路径即可。

    **注意**：如果策略文件夹下已存在 `merged_ckpt_strategy.ckpt` 且仍传入文件夹路径，脚本会首先删除旧的 `merged_ckpt_strategy.ckpt`，再合并生成新的 `merged_ckpt_strategy.ckpt` 以用于权重转换。因此，请确保该文件夹具有足够的写入权限，否则操作将报错。
- **src_ckpt_path_or_dir**：源权重的路径。如果为分布式权重，请填写源权重所在文件夹的路径，源权重应按 `model_dir/rank_x/xxx.ckpt` 格式存放，并将文件夹路径填写为 `model_dir`。若源权重为完整权重，则填写完整权重的绝对路径。
- **dst_ckpt_dir**：目标权重的保存路径，需为自定义的空文件夹路径。目标权重将按 `model_dir/rank_x/xxx.ckpt` 格式保存。
- **prefix**：目标权重文件的命名前缀，默认值为 "checkpoint_"，即目标权重将按照 `model_dir/rank_x/checkpoint_x.ckpt` 格式保存。
- **lora_scaling**：LoRA 权重的合并系数，默认为 `lora_alpha/lora_rank`，这两个参数即为 LoRA 模型配置时的参数，需自行计算。

### 示例

#### 场景一：包含 LoRA 参数的完整权重

如果合并前的权重是完整的权重文件，可以按照以下方式填写参数（直接输入完整权重的路径）：

```shell
python mindformers/tools/transform_ckpt_lora.py \
  --src_ckpt_path_or_dir .../xxx/xxx.ckpt \
  --dst_ckpt_dir dst_ckpt_dir \
  --prefix "checkpoint_" \
  --lora_scaling lora_alpha/lora_rank
```

#### 场景二：包含 LoRA 参数的分布式权重

如果合并前的权重是分布式的权重文件，可以按照以下方式填写参数（需输入分布式权重文件夹路径和分布式策略文件夹路径），最后得到的权重会自动合并为完整的权重文件：

```shell
python mindformers/tools/transform_ckpt_lora.py \
  --src_ckpt_strategy .../xxx/mindformers/output/strategy/ \
  --src_ckpt_path_or_dir .../xxx/model_dir \
  --dst_ckpt_dir dst_ckpt_dir \
  --prefix "checkpoint_" \
  --lora_scaling lora_alpha/lora_rank
```

## Safetensors权重离线合并

### 使用说明

使用MindSpore Transformers提供的[safetensors权重合并脚本](https://gitee.com/mindspore/mindformers/blob/r1.5.0/toolkit/safetensors/unified_safetensors.py)，按照如下方式进行safetensors权重合并。

```shell
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs src_strategy_path_or_dir \
  --mindspore_ckpt_dir mindspore_ckpt_dir\
  --output_dir output_dir \
  --file_suffix "1_1" \
  --has_redundancy has_redundancy
```

#### 参数说明

- **src_strategy_dirs**：源权重对应的分布式策略文件路径，通常在启动训练任务后默认保存在 `output/strategy/` 目录下。分布式权重需根据以下情况填写：
    - **源权重开启了流水线并行**：权重转换基于合并的策略文件，填写分布式策略文件夹路径。脚本会自动将文件夹内的所有 `ckpt_strategy_rank_x.ckpt` 文件合并，并在文件夹下生成 `merged_ckpt_strategy.ckpt`。如果已经存在 `merged_ckpt_strategy.ckpt`，可以直接填写该文件的路径。
    - **源权重未开启流水线并行**：权重转换可基于任一策略文件，填写任意一个 `ckpt_strategy_rank_x.ckpt` 文件的路径即可。

    **注意**：如果策略文件夹下已存在 `merged_ckpt_strategy.ckpt` 且仍传入文件夹路径，脚本会首先删除旧的 `merged_ckpt_strategy.ckpt`，再合并生成新的 `merged_ckpt_strategy.ckpt` 以用于权重转换。因此，请确保该文件夹具有足够的写入权限，否则操作将报错。
- **mindspore_ckpt_dir**：分布式权重路径，请填写源权重所在文件夹的路径，源权重应按 `model_dir/rank_x/xxx.safetensors` 格式存放，并将文件夹路径填写为 `model_dir`。
- **output_dir**：目标权重的保存路径，默认值为 "/new_llm_data/******/ckpt/nbg3_31b/tmp"，即目标权重将放置在 `/new_llm_data/******/ckpt/nbg3_31b/tmp` 目录下。
- **file_suffix**：目标权重文件的命名后缀，默认值为 "1_1"，即目标权重将按照 `*1_1.safetensors` 格式查找。
- **has_redundancy**：合并的权重是否是去除冗余的权重，默认为 `True`。
- **filter_out_param_prefix**：合并权重时可自定义过滤掉部分参数，过滤规则以前缀名匹配。如优化器参数"adam_"。
- **max_process_num**：合并最大进程数。默认值：64。

### 示例

#### 场景一：去除冗余的safetensors权重

如果合并去除冗余的safetensors权重，可以按照以下方式填写参数：

```shell
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs src_strategy_path_or_dir \
  --mindspore_ckpt_dir mindspore_ckpt_dir\
  --output_dir output_dir \
  --file_suffix "1_1" \
  --has_redundancy True
```

#### 场景二：不去除冗余的safetensors权重

如果合并非去除冗余的safetensors权重，可以按照以下方式填写参数：

```shell
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs src_strategy_path_or_dir \
  --mindspore_ckpt_dir mindspore_ckpt_dir\
  --output_dir output_dir \
  --file_suffix "1_1" \
  --has_redundancy False
```

#### 场景三：过滤Adam优化器的safetensors权重

如果合并过滤Adam优化器的safetensors权重，可以按照以下方式填写参数：

```shell
python toolkit/safetensors/unified_safetensors.py \
  --src_strategy_dirs src_strategy_path_or_dir \
  --mindspore_ckpt_dir mindspore_ckpt_dir\
  --output_dir output_dir \
  --file_suffix "1_1" \
  --filter_out_param_prefix "adam_"
```