# Ckpt权重

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/feature/ckpt.md)

## 概述

ckpt是深度学习框架中用于保存模型训练状态的通用文件格式，包含模型参数、优化器状态和训练进度等信息，主要用于恢复训练或微调模型，本文主要介绍MindSpore Transformers如何支持该文件格式的转换和切分。

> 已计划日落ckpt格式，使用权重更推荐使用safetensors格式。Safetensors 是 Huggingface 推出的一种可靠、易移植的机器学习模型存储格式，用于安全地存储Tensor，而且存储速度较快。详细参考文档[Safetensors权重](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/safetensors.html)。

## 权重格式转换

### 概述

MindSpore Transformers提供了统一的权重转换工具，能够将模型权重在HuggingFace所使用的格式与MindSpore Transformers所使用的格式之间相互转换。这可以帮助用户：

- 将HuggingFace权重转换为MindSpore Transformers权重，在MindSpore Transformers上进行微调、测评或推理。
- 把使用MindSpore Transformers训练或微调得到的权重转换为HuggingFace权重，并在其他框架上使用。

### 转换步骤

要进行权重转换，首先请将待转换模型的HuggingFace仓库完整克隆到本地，然后执行`mindformers/convert_weight.py`脚本。该脚本能够自动将HuggingFace的模型权重文件转换为适用于MindSpore Transformers的权重文件。如若希望将MindSpore Transformers权重转为HuggingFace权重，请将`reversed`设置为`True`。

```shell
python convert_weight.py [-h] --model MODEL [--reversed] --input_path INPUT_PATH  --output_path OUTPUT_PATH [--dtype DTYPE] [--n_head N_HEAD] [--hidden_size HIDDEN_SIZE] [--layers LAYERS] [--is_pretrain IS_PRETRAIN] [--telechat_type TELECHAT_TYPE]
```

#### 参数说明

- model：模型名称。
- reversed：将MindSpore Transformers权重转换为HuggingFace权重。
- input_path：HuggingFace权重文件夹的路径，指向已下载的权重文件。
- output_path：转换后MindSpore Transformers权重文件的保存路径。
- dtype：转换后的权重数据类型。
- n_head：只对BLOOM模型生效，使用`bloom_560m`时请设为`16`，使用`bloom_7.1b`时请设为`32`。
- hidden_size：只对BLOOM模型生效，使用`bloom_560m`时请设为`1024`，使用`bloom_7.1b`时请设为`4096`。
- layers：只对GPT2和WizardCoder模型生效，模型被转换的层数。
- is_pretrain：只对Swin模型生效，转换预训练权重。
- telechat_type：只对TeleChat模型生效，TeleChat模型的版本。

### 转换示例

假设用户已经下载了[Llama2模型的权重](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama2.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E4%B8%8B%E8%BD%BD)，并保存在路径`/home/user/torch_weights`中，用户希望将其转换为MindSpore Transformers权重并保存在路径`/home/user/ms_weights`中，可以使用以下命令：

```bash
python convert_weight.py --model llama --input_path /home/user/torch_weights --output_path /home/user/ms_weights/llama.ckpt
```

通过以上步骤，可将HuggingFace权重成功转换为MindSpore Transformers权重，方便在MindSpore Transformers中继续模型训练或推理。

### 已支持模型

| 参数取值      | 支持模型                                      |
|-----------|-------------------------------------------|
| llama     | Llama2、Llama3、Llama3.1、CodeLlama          |
| baichuan2 | Baichuan2                                 |
| glm-n     | GLM2、GLM3、GLM3-32K、GLM4                   |
| cogvlm2   | CogVLM2-Video、CogVLM2-Image               |
| qwen      | Qwen、Qwen1.5、Qwen2                        |
| qwenvl    | QwenVL                                    |
| internlm  | InternLM                                  |
| internlm2 | InternLM2                                 |
| yi        | Yi                                        |
| mixtral   | Mixtral                                   |
| deepseek  | DeepSeekCoder、DeepSeekCoder1.5、DeepSeekV2 |
| gpt       | GPT2                                      |
| whisper   | Whisper                                   |

### 未支持模型权重转换开发

1. 在扩展模型目录下新增`convert_weight.py`及`convert_reversed.py`文件。
2. 在文件中分别编写`convert_pt_to_ms`及`convert_ms_to_pt`权重转换函数，函数参数为`input_path`、`output_path`、`dtype`及额外参数`**kwargs`。
3. 在MindSpore Transformers代码根目录下`convert_weight.py`文件中的`convert_map`和`reversed_convert_map`字典中加入扩展模型名称及转换函数引入路径。
4. 在`main`函数中通过调用`parser.add_argument()`方法新增额外参数。

### 模型权重转换开发示例

此处以Llama为例。如若希望转换HuggingFace权重至MindSpore Transformers权重，需在[convert_weight.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/convert_weight.py)内定义`convert_pt_to_ms`函数：

```python
def convert_pt_to_ms(input_path, output_path, dtype=None, **kwargs):
    """convert hf weight to ms."""
    print(f"Trying to convert huggingface checkpoint in '{input_path}'.", flush=True)
    try:
        from transformers import LlamaForCausalLM
    except:
        raise ImportError(f"Failed to load huggingface checkpoint. Please make sure transformers is available.")

    try:
        model_hf = LlamaForCausalLM.from_pretrained(os.path.dirname(input_path))
    except Exception as e:
        print(f"Do not find huggingface checkpoint in '{os.path.dirname(input_path)}', Error {e.message}.", flush=True)
        return False
    ckpt_list = []
    for name, value in model_hf.state_dict().items():
        name = name_replace(name)
        if name == 'norm.weight':
            name = 'norm_out.weight'
        if name[:7] == 'layers.':
            name = name[7:]

        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        ckpt_list.append({'name': name, 'data': pt2ms(value, dtype)})

    ms.save_checkpoint(ckpt_list, output_path)
    print(f"\rConvert huggingface checkpoint finished, the mindspore checkpoint is saved in '{output_path}'.",
          flush=True)
    return True
```

而若是希望转换MindSpore Transformers权重至HuggingFace权重，则需在[convert_reversed.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/models/llama/convert_reversed.py)内定义`convert_ms_to_pt`函数：

```python
def convert_ms_to_pt(input_path, output_path, dtype=None, **kwargs):
    """convert ms weight to hf."""
    print(f"Trying to convert mindspore checkpoint in '{input_path}'.", flush=True)
    model_ms = ms.load_checkpoint(input_path)

    state_dict = {}
    for name, value in model_ms.items():
        name = name_replace(name)
        print(f'\rprocessing parameter: {name} {value.shape}     ', end='', flush=True)
        if is_lora_param(name):
            name = name.replace('.tk_delta_lora_a', '.lora_A.weight')
            name = name.replace('.tk_delta_lora_b', 'lora_B.weight')
        state_dict[name] = ms2pt(value, dtype)

    torch.save(state_dict, output_path)
    print(f"\rConvert mindspore checkpoint finished, the huggingface checkpoint is saved in '{output_path}'.",
          flush=True)
    return True
```

## 权重切分与合并

### 概述

在当前的分布式训练和推理环境中，当预训练权重与分布式策略不匹配时，需要对预训练权重进行转换，以适应相应的分布式策略。为满足不同场景下的权重转换需求，MindSpore Transformers提供了一套权重转换工具。该工具支持单卡权重切分为多卡权重、多卡权重之间的转换、多卡权重合并为单卡权重。用户可根据具体需求选择[自动转换](#自动转换)或[离线转换](#离线转换)，帮助模型在不同分布式场景之间快速切换。

此外，MindSpore Transformers还支持[LoRA权重的合并](#lora权重合并)，方便用户部署使用LoRA微调后的模型。

### 自动转换

模型加载权重时，自动转换功能可以自动检测权重与当前模型分布式切分策略之间的匹配情况，如果不匹配，自动进行权重转换，无需用户手动干预。

#### 参数说明

**自动权重转换**相关`yaml`文件参数说明如下：

| 参数名称              | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------- |------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| load_checkpoint     | 预加载权重的绝对路径或文件夹路径。<br> - 如果是完整权重，则填写绝对路径；<br> - 如果是分布式权重，则填写文件夹路径，分布式权重须按照`model_dir/rank_x/xxx.ckpt`格式存放，文件夹路径填写为`model_dir`。<br>**如果rank_x文件夹下存在多个ckpt，将会使用文件名默认排序最后的ckpt文件用于转换。**                                                                                                                                                                                                                                                |
| src_strategy_path_or_dir        | 预加载权重对应的[分布式策略文件](#离线转换配置说明)路径。<br> - 如果预加载权重是完整权重，则**不填写**；<br> - 如果预加载权重是分布式权重，且预加载权重保存时使用了流水线并行，则填写**合并的策略文件路径**或**分布式策略文件夹路径**；<br> - 如果预加载权重是分布式权重，且预加载权重保存时未使用流水线并行，则填写任一**ckpt_strategy_rank_x.ckpt**路径；                                                                                                                                                                                                                     |
| auto_trans_ckpt     | 权重自动转换开关，为True开启，默认False。                                                                                                                                                                                                                                                                                                                                                                                                          |
| transform_process_num | 权重自动转换使用的进程数，默认为1。<br> - 如果transform_process_num = 1，使用**单进程转换**，转换时只有rank_0负责权重转换，其他进程等待rank_0转换结束；<br> - 如果transform_process_num > 1，使用**多进程转换**，比如8卡任务，transform_process_num=2时，转换时rank_0负责rank_0/1/2/3切片权重的转换，rank_4负责rank_4/5/6/7切片权重的转换，其他进程等待rank_0/4转换结束；<br>**注意**：<br> 1. transform_process_num越大，转换时间越短，**转换所占用的host内存越大**；当出现host侧内存不足时，需要减少transform_process_num。<br> 2. transform_process_num必须能够整除NPU卡数，且最大不得超过NPU卡数。 |
| transform_by_rank   | 是否使用mindspore.transform_checkpoint_by_rank接口做权重转换。<br> - transform_process_num > 1时，自动设置为`True`；<br> - transform_process_num = 1时，如果目标权重为分布式权重，则循环调用mindspore.transform_checkpoint_by_rank串行转换每一个rank切片权重。<br>- transform_process_num = 1时，如果目标权重为完整权重，则自动设置为`False`，使用mindspore.transform_checkpoints接口做权重转换；                                                                                                                     |

#### 不同场景下yaml配置说明

**单卡权重切分为多卡权重**

```yaml
# load_checkpoint: 设置为预训练权重文件路径
load_checkpoint: "/worker/llama3_8b/llama3_8b.ckpt"

# auto_trans_ckpt: 开启自动转换
auto_trans_ckpt: True
```

**多卡权重之间的转换**

```yaml
# load_checkpoint: 设置为多卡权重文件夹路径
load_checkpoint: "/worker/checkpoint/llama3-8b-2layer-dp2mp2pp2"

# src_strategy_path_or_dir: 设置为分布式策略文件路径
src_strategy_path_or_dir: "/worker/checkpoint/llama3-8b-2layer-dp2mp2pp2/strategy/merged_ckpt_strategy.ckpt"

# auto_trans_ckpt: 开启自动转换
auto_trans_ckpt: True
```

**多卡权重合并为单卡权重**

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

**开启多进程转换（可选）**

```yaml
# transform_process_num: 设置参与转换的进程数量
transform_process_num: 2
```

#### 注意事项

- **多进程转换**：配置`transform_process_num`参数以开启多进程转换，但需注意内存占用。如果发生内存溢出，建议降低进程数量。

- **自动权重转换**：开启自动转换后，系统将删除`output`目录下的旧`strategy`和`transformed_checkpoint`文件夹，并保存当前任务的输出结果。建议在转换任务结束后，将`strategy`和`transformed_checkpoint`文件夹移动到自定义目录，以避免后续操作中被误删。

- **分布式策略文件保存**：分布式策略文件将保存在`output/strategy`文件夹下。如果开启了**流水线并行**，系统会自动合并所有的`ckpt_strategy_rank_x.ckpt`文件，生成`merged_ckpt_strategy.ckpt`。如果未开启流水线并行，则不会进行合并操作。

### 离线转换

离线转换功能旨在满足用户手动转换权重的需求。通过离线转换，用户可以在独立的环境中进行模型权重的转换操作。离线转换支持多种权重转换场景，包括单卡权重切分为多卡权重、多卡权重之间的转换、多卡权重合并为单卡权重。

用户在使用离线转换时，可以根据具体需求手动配置转换参数，确保转换过程灵活且可控，尤其适用于在严格控制的计算环境中进行模型部署和优化的场景。

#### 参数说明

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

#### 离线转换配置说明

**生成分布式策略**

MindSpore每次运行分布式任务后都会在`output/strategy`文件夹下生成对应卡数的分布式策略文件（ckpt格式），可以在离线权重转换中使用。

如果当前没有分布式策略文件，可以通过这种方式快速生成：在原有分布式训练/推理任务的基础上，在yaml配置文件中设置`only_save_strategy:True`来生成策略文件。设置之后任务会在生成分布式策略文件后立即停止，而不会实际执行训练或推理。

**单进程转换**

使用[mindformers/tools/ckpt_transform/transform_checkpoint.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/ckpt_transform/transform_checkpoint.py)对载入权重进行单进程转换。

**运行命令**：

```shell
python transform_checkpoint.py \
  --src_checkpoint /worker/checkpoint/llama3-8b-2layer/rank_0/llama3_8b.ckpt \
  --dst_checkpoint /worker/transform_ckpt/llama3_8b_1to8/ \
  --dst_strategy /worker/mindformers/output/strategy/
```

**多进程转换**

使用[mindformers/tools/ckpt_transform/transform_checkpoint.sh](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/ckpt_transform/transform_checkpoint.sh)对载入权重进行多进程转换。

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

- 使用[transform_checkpoint.sh](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/ckpt_transform/transform_checkpoint.sh)脚本时，参数`8`表示目标设备数，参数`2`表示使用2个进程进行转换。

### 特殊场景

#### 物理机多机多卡训练

大规模模型通常需要通过多台服务器组成的集群进行训练。在这种多机多卡的场景下，如果服务器之间存在共享盘，则可以使用自动转换功能，否则只能使用离线转换。下面以两台服务器、16卡训练为例进行说明。

**场景一：服务器之间有共享盘**

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

  使用[mindformers/scripts/msrun_launcher.sh](https://gitee.com/mindspore/mindformers/blob/dev/scripts/msrun_launcher.sh)进行任务启动。

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

**场景二：服务器之间无共享盘**

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

  在保存有所有策略文件的服务器上，使用[mindformers/tools/ckpt_transform/transform_checkpoint.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/ckpt_transform/transform_checkpoint.py)进行离线权重转换。

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

#### ModelArts 训练

在 ModelArts 环境中进行训练与物理机上的多机多卡训练类似，同样支持开启权重自动转换。用户可以通过在训练作业的超参数中配置`auto_trans_ckpt=True`来启用自动权重转换，并通过设置`transform_process_num > 1`来开启多进程转换。

**注意**：如果 ModelArts 资源池中的服务器节点NPU卡数不是8，则需要额外配置`npu_num_per_node=节点NPU卡数`。例如，如果每个节点配有16个NPU，则应设置`npu_num_per_node=16`。

### LoRA权重合并

#### 概述

LoRA（Low-Rank Adaptation）的基本原理是对原始模型的参数进行低秩重参数化。合并LoRA权重的核心过程是将 LoRA 分支的参数进行计算，并叠加到对应的模型参数中，使最终得到的权重文件的参数列表与原始模型一致，不包含额外的LoRA参数。这一操作不会对推理结果产生任何影响，因此合并后的模型在推理时依然能够保持与原始模型一致的性能。
有关 LoRA 的详细原理和实现，请参阅以下资源：

- 论文: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- GitHub: [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)

#### 使用说明

使用MindSpore Transformers提供的[LoRA权重合并脚本](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/transform_ckpt_lora.py)，按照如下方式进行LoRA权重合并。

```shell
python mindformers/tools/transform_ckpt_lora.py \
  --src_ckpt_strategy src_strategy_path_or_dir \
  --src_ckpt_path_or_dir src_ckpt_path_or_dir \
  --dst_ckpt_dir dst_ckpt_dir \
  --prefix "checkpoint_" \
  --lora_scaling lora_alpha/lora_rank
```

**参数说明**

- **src_ckpt_strategy**：源权重对应的分布式策略文件路径，通常在启动训练任务后默认保存在 `output/strategy/` 目录下。如果源权重为完整权重，则无需填写此参数；如果为分布式权重，需根据以下情况填写：
    - **源权重开启了流水线并行**：权重转换基于合并的策略文件，填写分布式策略文件夹路径。脚本会自动将文件夹内的所有 `ckpt_strategy_rank_x.ckpt` 文件合并，并在文件夹下生成 `merged_ckpt_strategy.ckpt`。如果已经存在 `merged_ckpt_strategy.ckpt`，可以直接填写该文件的路径。
    - **源权重未开启流水线并行**：权重转换可基于任一策略文件，填写任意一个 `ckpt_strategy_rank_x.ckpt` 文件的路径即可。

    **注意**：如果策略文件夹下已存在 `merged_ckpt_strategy.ckpt` 且仍传入文件夹路径，脚本会首先删除旧的 `merged_ckpt_strategy.ckpt`，再合并生成新的 `merged_ckpt_strategy.ckpt` 以用于权重转换。因此，请确保该文件夹具有足够的写入权限，否则操作将报错。
- **src_ckpt_path_or_dir**：源权重的路径。如果为分布式权重，请填写源权重所在文件夹的路径，源权重应按 `model_dir/rank_x/xxx.ckpt` 格式存放，并将文件夹路径填写为 `model_dir`。若源权重为完整权重，则填写完整权重的绝对路径。
- **dst_ckpt_dir**：目标权重的保存路径，需为自定义的空文件夹路径。目标权重将按 `model_dir/rank_x/xxx.ckpt` 格式保存。
- **prefix**：目标权重文件的命名前缀，默认值为 "checkpoint_"，即目标权重将按照 `model_dir/rank_x/checkpoint_x.ckpt` 格式保存。
- **lora_scaling**：LoRA 权重的合并系数，默认为 `lora_alpha/lora_rank`，这两个参数即为 LoRA 模型配置时的参数，需自行计算。

#### 示例

**场景一：包含 LoRA 参数的完整权重**

如果合并前的权重是完整的权重文件，可以按照以下方式填写参数（直接输入完整权重的路径）：

```shell
python mindformers/tools/transform_ckpt_lora.py \
  --src_ckpt_path_or_dir .../xxx/xxx.ckpt \
  --dst_ckpt_dir dst_ckpt_dir \
  --prefix "checkpoint_" \
  --lora_scaling lora_alpha/lora_rank
```

**场景二：包含 LoRA 参数的分布式权重**

如果合并前的权重是分布式的权重文件，可以按照以下方式填写参数（需输入分布式权重文件夹路径和分布式策略文件夹路径），最后得到的权重会自动合并为完整的权重文件：

```shell
python mindformers/tools/transform_ckpt_lora.py \
  --src_ckpt_strategy .../xxx/mindformers/output/strategy/ \
  --src_ckpt_path_or_dir .../xxx/model_dir \
  --dst_ckpt_dir dst_ckpt_dir \
  --prefix "checkpoint_" \
  --lora_scaling lora_alpha/lora_rank
```
