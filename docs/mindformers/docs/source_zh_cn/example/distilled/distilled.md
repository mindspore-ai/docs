# 使用DeepSeek-R1进行模型蒸馏的实践案例

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/distilled/distilled.md)

本案例参考OpenR1-Qwen-7B，旨在指导用户基于MindSpore框架和MindSpore Transformers大模型套件，使用DeepSeek-R1对Qwen2.5-Math-7B模型进行知识蒸馏和微调，以提升其在数学推理任务上的性能。案例涵盖了从环境配置、数据生成、预处理到模型微调和推理测试的完整流程。通过以下步骤，您可以了解如何利用DeepSeek-R1生成推理数据、过滤错误数据、处理数据集，并最终对模型进行微调以解决复杂的数学问题。

蒸馏流程：

![蒸馏流程](./images/distilled_process.png)

更多信息请参考[DeepSeek-R1-Distill-Qwen-7B](https://hf-mirror.com/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

## 1. 前提准备

### 1.1 环境

安装方式请参考[MindSpore Transformers安装指南](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/installation.html)。

并将本案例的[distilled](https://gitee.com/mindspore/docs/tree/master/docs/mindformers/docs/source_zh_cn/example/distilled/distilled)文件夹，复制到MindSpore Transformers源码根目录下。

最后得到的目录结构如下：

```bash
mindformers
├── ...
└── distilled
    ├── data_process_handling.yaml  # 数据集处理配置文件
    ├── data_process_packing.yaml   # 数据集packing配置文件
    ├── finetune_qwen_2_5_7b.yaml   # 微调配置文件
    ├── generate_reasoning.py       # 生成CoT数据脚本
    └── reject_sampling.py          # 拒绝采样脚本
```

> 本案例的指令均在MindSpore Transformers源码根目录下执行。

### 1.2 模型

本次微调使用的模型为Qwen2.5-Math-7B-Instruct，可以在[魔乐社区](https://modelers.cn/models/MindSpore-Lab/Qwen2.5-Math-7B-Instruct)下载。

### 1.3 数据集

本案例提供三种数据集的准备方式：

- **从零开始生成数据集**：适合希望自定义数据集或深入了解数据生成流程的用户。包括从种子数据集生成CoT数据和拒绝采样。请从[1.3.1 从零开始生成数据集](#131-从零开始生成数据集)开始。
- **使用OpenR1-Math-220K数据集**：

    - **选项1: 使用原始数据离线处理**：适合需要自定义数据处理或学习处理流程的用户。包括预处理和Packing。请从[选项1: 使用原始数据离线处理](#选项-1-使用原始数据离线处理)开始。
    - **选项2: 使用已处理好的数据**：适合希望快速开始训练的用户。案例提供预处理好的OpenR1-Math-220K数据集。请从[选项2: 使用已处理好的数据](#选项-2-使用完成转换的数据)开始。

#### 1.3.1 从零开始生成数据集

**适用场景**：适合希望自定义数据集或学习数据生成流程的用户。  

> 生成数据集流程仅作为示例，如需生成高质量数据集，建议参考[OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k)的数据集生成流程。

1. 安装依赖

    执行以下命令安装所需依赖：

    ```shell
    pip install datasets tqdm aiofiles aiohttp uvloop math_verify
    ```

2. 本地部署Deepseek-R1

    参考[MindSpore-Lab/DeepSeek-R1 | 魔乐社区](https://modelers.cn/models/MindSpore-Lab/DeepSeek-R1)在本地部署DeepSeek-R1推理服务，或是使用公开的API服务。

3. 生成数据

    **目标**：利用DeepSeek-R1模型为数学问题生成Chain-of-Thought（CoT）推理数据，用于后续的数据蒸馏。

    首先需要在脚本`generate_reasoning.py`中修改API_KEY。

    ```python
    API_KEY = "your_api_key_here"
    ```

    执行以下命令调用推理服务的接口，使用种子数据集中的问题，生成CoT数据：

    ```shell
    python distilled/generate_reasoning.py \
        --model DeepSeek-R1 \
        --dataset-name AI-MO/NuminaMath-1.5 \
        --output-file /path/to/numinamath_r1_generations.jsonl \
        --prompt-column problem \
        --uuid-column problem \
        --api-addr api.host.name \
        --num-generations 2 \
        --max-tokens 16384 \
        --max-concurrent 100
    ```

    - **作用**：调用DeepSeek-R1推理服务，基于[AI-MO/NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)数据集中的数学问题（`problem`列）生成推理路径。
    - **参数说明**：

        - **`--model`**: 推理服务的模型名，需要和服务化配置文件 `config.json` 中的 `modelName` 一致。
        - **`--dataset-name`**：种子数据集名称，配置为HuggingFace Datasets名称或本地的数据集路径。
        - **`--output-file`**：输出CoT数据文件的文件名。
        - **`--prompt-column`**：种子数据集中提示词的列名，使用此列的数据进行CoT数据生成。
        - **`--uuid-column`**：种子数据集中uuid的列名，使用此列计算哈希值去重数据。
        - **`--api-addr`**：推理服务api的地址，配置为 `ip:port` 。
        - **`--num-generations`**：对于种子数据集中每个问题生成CoT数据的数量。
        - **`--max-tokens`**：生成的CoT数据的最大Token数。
        - **`--max-concurrent`**：请求的最大并发数量。

1. 拒绝采样

    **目标**：过滤掉推理数据中的错误或不准确的CoT数据，确保数据质量。

    ``` shell
    python distilled/reject_sampling.py \
        --src /path/to/numinamath_r1_generations.jsonl \
        --dst /path/to/numinamath_r1_generations_filtered.jsonl
    ```

    - **作用**：使用`math_verify`库验证`numinamath_r1_generations.jsonl`中的推理路径，剔除错误的CoT数据。
    - **参数说明**：

        - **`--src`**：输入的CoT数据文件路径。
        - **`--dst`**：输出的过滤后的CoT数据文件路径。

2. 数据集预处理

    跳转到[选项-1-使用原始数据离线处理](#选项-1-使用原始数据离线处理)的中的**步骤一**，并将生成的CoT数据转换为MindSpore Transformers支持的格式。

    **此时的数据集格式为jsonl格式，和原始数据集的parquet格式不一致，并且`data_files`中只包含一个`numinamath_r1_generations_filtered.jsonl`文件。按照以下格式修改配置文件`data_process_handling.yaml`**：

    ```yaml
    train_dataset:
    ...
    data_loader:
        ...
        path: "json"
        data_files:
            ["/path/to/numinamath_r1_generations_filtered.jsonl"]
        ...
    ```

#### 1.3.2 使用OpenR1-Math-220K数据集

**适用场景**：适合希望使用高质量预蒸馏数据集进行微调的用户。

如果使用OpenR1-Math-220K数据集（已经过DeepSeek-R1蒸馏）进行微调，我们提供[详细制作流程](#选项-1-使用原始数据离线处理)以及[转换后的数据集](#选项-2-使用完成转换的数据)。

##### 选项 1: 使用原始数据离线处理

首先在HuggingFace上下载[OpenR1-Math-220K](https://huggingface.co/datasets/open-r1/OpenR1-Math-220K)原始数据集。

步骤一、**数据集预处理**

**目标**：将原始数据集（例如OpenR1-Math-220K）转换为适合MindSpore Transformers微调的格式。

首先需要修改数据集处理的配置文件`data_process_handling.yaml`：

1. 将MindSpore Transformers源码根目录下的`research/qwen2_5/qwen2_5_tokenizer.py`文件复制到`distilled`目录下。

    ```bash
    cp research/qwen2_5/qwen2_5_tokenizer.py distilled/
    ```

2. 修改数据集文件路径：将`data_files`中的路径替换为原始数据集的路径。每一个parquet文件都需要在这里列出。
    - 例如：`["/path/to/data1.parquet", "/path/to/data2.parquet", ...]`。
3. 修改tokenizer的路径：将`vocab_file`和`merges_file`替换为Qwen2.5-7B-Instruct模型的**词表文件**和**merges文件**的路径。

    ```yaml
    train_dataset:
    input_columns: &input_columns ["input_ids", "labels"]
    data_loader:
        ...
        data_files:
            ["/path/to/data1.parquet", "/path/to/data2.parquet", ...]   # 数据集文件路径
        handler:
        - type: OpenR1Math220kDataHandler
            ...
          tokenizer:
            ...
            vocab_file: "/path/to/vocab.json"       # 词表文件路径
            merges_file: "/path/to/merges.txt"      # merges文件路径
            chat_template: ...
        ...
    ```

    在MindSpore Transformers源码根目录下执行以下数据预处理脚本：

    ```shell
    python toolkit/data_preprocess/huggingface/datasets_preprocess.py \
        --config distilled/data_process_handling.yaml \
        --save_path /path/to/handled_data \
        --register_path distilled/
    ```

    - **作用**：将原始数据集转换为MindSpore Transformers支持的格式。
    - **参数说明**：

        - **`--config`**：数据预处理的配置文件路径。
        - **`--save_path`**：转换后数据集的保存文件夹路径。
        - **`--register_path`**：注册路径，为当前目录下的`distilled/`文件夹。

步骤二、**数据集Packing**

MindSpore Transformers已经支持数据集packing机制，减少微调所需要的时间。
数据集packing的配置文件放在/dataset/packing目录下。其中，需要将`path`修改成`handled_data`的路径，

```yaml
# dataset
train_dataset:
  data_loader:
    ...
    path: /path/to/handled_data # 转换后数据集的保存文件夹
```

并在MindSpore Transformers源码根目录下执行如下脚本：

```shell
python toolkit/data_preprocess/huggingface/datasets_preprocess.py \
    --config distilled/data_process_packing.yaml \
    --save_path /path/to/packed_data \
    --register_path distilled
```

- **作用**：将处理好的数据集进行packing，减少微调时的数据加载时间。
- **参数说明**：

    - **`--config`**：数据集packing的配置文件路径。
    - **`--save_path`**：packing后数据集的保存路径
    - **`--register_path`**：注册数据集的路径。

最后在`packed_data`中可以找到处理后的数据集，格式为arrow。

更多数据集处理的教程请参考[MindSpore Transformers官方文档-数据集](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/dataset.html#%E8%87%AA%E5%AE%9A%E4%B9%89%E6%95%B0%E6%8D%AEhandler)。

##### 选项 2: 使用完成转换的数据

我们在[魔乐社区](https://modelers.cn/models/MindSpore-Lab/OpenR1-Qwen-7B/tree/main/dataset/packing)提供packing处理后可以直接用于模型训练的数据，格式为arrow。此时[#1.4 YAML配置](#14-yaml配置)中的`path`需要修改为下载后的数据集路径。

```yaml
train_dataset:
  ...
  data_loader:
    ...
    path: "/path/to/OpenR1-Qwen-7B/dataset/packing/"
```

### 1.4 YAML配置

微调配置文件`finetune_qwen_2_5_7b.yaml`，需要根据实际情况修改，具体如下：

```yaml
seed: 42
output_dir: './output'
load_checkpoint: "/path/to/Qwen2.5-Math-7B-Instruct" # 权重文件夹路径，根据实际情况修改
load_ckpt_format: 'safetensors'
auto_trans_ckpt: True
only_save_strategy: False
resume_training: False
run_mode: 'finetune'
...
train_dataset: &train_dataset
  input_columns: &input_columns ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
  divisor: 32
  remainder: 1
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 2
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  dynamic_batch: True
  pad_token_id: 151643
  data_loader:
    type: CommonDataLoader
    shuffle: True
    split: "train"
    load_func: "load_from_disk"
    path: "/path/to/packed_data" # packing处理后的数据集文件夹路径
......
```

其余参数配置的解释可以参考[MindSpore Transformers官方文档-SFT微调](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/guide/supervised_fine_tuning.html)。

## 2. 启动微调

设置如下环境变量防止OOM：

```bash
export ACLNN_CACHE_LIMIT=10 # CANN 缓存限制
export MS_DEV_RUNTIME_CONF="aclnn_cache_queue_length:128" # MS缓存队列长度建议设置成128，设置过大内存容易OOM，设置越小性能越差
```

在MindSpore Transformers目录下执行如下命令行启动微调：

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py --config distilled/finetune_qwen_2_5_7b.yaml --run_mode finetune" 8
```

日志记录在`output/msrun_log`目录下，例如可以通过`tail -f output/msrun_log/worker_7.log`指令查看worker 7的日志信息。
微调完成后，输出的`safetensors`权重文件在`output/checkpoint`目录下。

更多safetensors权重的内容请参考[MindSpore Transformers官方文档-Safetensors权重](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/feature/safetensors.html)。

## 3. 执行推理

若想使用微调后的权重进行推理，可以参考[Qwen2.5-Math-7B-Instruct](https://modelers.cn/models/MindSpore-Lab/Qwen2.5-Math-7B-Instruct)中的推理部分，但需要修改[run_qwen2_5.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/research/qwen2_5/run_qwen2_5.py)脚本中的system的提示词：

```python
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": input_prompt}
    ]
```

## 4. 评估结果

| Model                                   | MATH-500 |
|-----------------------------------------|:--------:|
| DeepSeek-Distill-Qwen-7B                | 91.6     |
| OpenR1-Qwen-7B (HuggingFace)            | 90.6     |
| OpenR1-Qwen-7B (MindSpore Transformers) | 90.0     |
| OpenThinker-7B                          | 89.6     |

> 上表第三行为本案例实验结果，该结果由本地实测得到。
