# 数据集

MindSpore Transformers 动态图（PyNative）模式目前支持多种类型的数据集加载方式，涵盖常用开源与自定义场景。具体包括：

- **Megatron 数据集**：支持加载符合 Megatron-LM 格式的数据集，适用于大规模语言模型的预训练任务。
- **HuggingFace 数据集**：兼容 HuggingFace `datasets` 库，方便直接调用社区中丰富的公开数据资源。
- **MindRecord 数据集**：MindRecord 是 MindSpore 提供的高效数据存储/读取模块，支持将不同公开数据集转换为 MindRecord 格式进行训练。

## 配置结构

动态图模式下，数据集配置位于 YAML 配置文件的 `train_dataset` 字段中，其整体结构如下：

```yaml
train_dataset:
  dataloader:
    type: BlendedMegatronDatasetDataLoader  # 或 HFDataLoader / MindDataset
    # ... 各 dataloader 类型特有的配置项
    column_names: ["input_ids", "labels"]   # 数据集返回的列名
    shuffle: false                          # 是否随机打乱数据集
    python_multiprocessing: false           # 是否使用 Python 多进程

  drop_remainder: true          # 是否丢弃最后一个不完整的 batch
  num_parallel_workers: 8       # 数据加载的并行工作线程数
  prefetch_size: 1              # 预取 batch 数量
  numa_enable: false            # 是否启用 NUMA 感知的数据加载
```

各字段说明如下：

| 参数名 | 说明 | 类型 |
|--------|------|:----:|
| `dataloader.type` | 数据加载器类型，可选值为 `BlendedMegatronDatasetDataLoader`、`HFDataLoader`、`MindDataset` | str |
| `dataloader.column_names` | 数据集返回的列名列表，默认值为 `["input_ids", "labels"]` | list |
| `dataloader.shuffle` | 是否对数据集进行随机打乱，默认值为 `false` | bool |
| `dataloader.python_multiprocessing` | 是否使用 Python 多进程，默认值为 `false` | bool |
| `drop_remainder` | 是否丢弃最后一个不完整的 batch，默认值为 `true` | bool |
| `num_parallel_workers` | 数据加载的并行工作线程数，默认值为 `8` | int |
| `prefetch_size` | 预取的 batch 数量，默认值为 `1` | int |
| `numa_enable` | 是否启用 NUMA 感知的数据加载，默认值为 `false` | bool |

## Megatron 数据集

Megatron 数据集是为大规模分布式语言模型预训练场景设计的一种高效数据格式，广泛应用于 Megatron-LM 框架。该数据集通常经过预处理，序列化为二进制格式（例如 `.bin` 或 `.idx` 文件），并配套特定索引机制，便于在分布式集群环境下高效并行加载与数据切分。

下面将分别介绍如何生成 `.bin` 或 `.idx` 文件以及在训练任务中使用 Megatron 数据集。

### 数据预处理

MindSpore Transformers 提供了数据预处理脚本 [preprocess_indexed_dataset.py](https://atomgit.com/mindspore/mindformers/blob/master/toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py)，用于将 `json` 格式的原始文本语料转换成 `.bin` 或 `.idx` 文件。如果用户的原始文本不是 `json` 格式，需要自行将数据处理成对应格式的文件。

下面是 `json` 格式文件的示例：

```json
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
...
```

各数据字段的说明如下：

| 字段名 | 说明 | 是否必须存在 |
|--------|------|:----:|
| text | 原始文本数据 | 是 |
| id | 数据的编号，按顺序排列 | 否 |
| src | 数据来源 | 否 |
| type | 数据的语言类型 | 否 |
| title | 数据标题 | 否 |

下面以 `wikitext-103` 数据集为例，介绍如何将数据集转换为 Megatron 数据集：

1. 下载 `wikitext-103` 数据集：[链接](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens)

2. 生成 `json` 格式数据文件

   `wikitext-103` 数据集原始文本如下：

   ```text
   = Valkyria Chronicles III =

   Valkyria Chronicles III is a tactical role-playing game developed by Sega for the PlayStation Portable.

   The game was released in Japan on January 27, 2011.

   = Gameplay =

   The game is similar to its predecessors in terms of gameplay...
   ```

   需要将原始文本处理成如下格式，并保存成 `json` 文件：

   ```json
   {"id": 0, "text": "Valkyria Chronicles III is a tactical role-playing game..."}
   {"id": 1, "text": "The game is similar to its predecessors in terms of gameplay..."}
   ...
   ```

3. 下载模型的词表文件

   由于不同模型对应不同的词表文件，因此需要下载对应训练模型的词表文件。这里以 `Qwen3-8B` 模型为例，下载 [tokenizer](https://huggingface.co/Qwen/Qwen3-8B) 以用于数据预处理。

4. 生成 `.bin` 或 `.idx` 数据文件

   执行数据预处理脚本 [preprocess_indexed_dataset.py](https://atomgit.com/mindspore/mindformers/blob/master/toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py) 可以将原始文本数据通过模型的 tokenizer 转换为对应的 token id。

   该脚本参数如下：

   | 参数名 | 说明 |
   |--------|------|
   | input | `json` 格式文件路径 |
   | output-prefix | `.bin` 或 `.idx` 数据文件格式的前缀 |
   | tokenizer-type | 模型使用的 tokenizer 类型 |
   | vocab-file | 模型使用的 tokenizer 文件（tokenizer.model/vocab.json）路径 |
   | merges-file | 模型使用的 tokenizer 文件（merge.txt）路径 |
   | tokenizer-file | 模型使用的 tokenizer 文件（tokenizer.json）路径 |
   | add_bos_token | 是否在句首中加入 `bos_token` |
   | add_eos_token | 是否在句尾中加入 `eos_token` |
   | eos_token | 代表 `eos_token` 的词元，默认为 `'</s>'` |
   | append-eod | 是否在文本的末尾添加一个 `eos_token` |
   | tokenizer-dir | 模型使用的 HuggingFaceTokenizer 的目录，仅在 `tokenizer-type`='HuggingFaceTokenizer' 时生效 |
   | trust-remote-code | 是否允许使用 Hub 上定义的 tokenizer 类，仅在 `tokenizer-type`='HuggingFaceTokenizer' 时生效 |
   | register_path | 选择外部 tokenizer 代码所在目录，仅在 `tokenizer-type`='AutoRegister' 时生效 |
   | auto_register | 选择外部 tokenizer 的导入路径，仅在 `tokenizer-type`='AutoRegister' 时生效 |

   `tokenizer-type` 的可选值为 `'HuggingFaceTokenizer'` 和 `'AutoRegister'`。其中，设置为 `'HuggingFaceTokenizer'` 时，transformers 库的 AutoTokenizer 类会使用本地 HuggingFace 仓库中的 tokenizer 进行实例化；设置为 `'AutoRegister'` 时，表示调用由 `register_path` 和 `auto_register` 参数指定的外部 tokenizer 类。

   以 [Deepseek-V3 仓库](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) 中的 [LlamaTokenizerFast](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/tokenizer_config.json) 和 [词表](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/tokenizer.json) 为例。如果本地不存在对应仓库，需要将配置文件（tokenizer_config.json）和词表文件（tokenizer.json）手动下载到本地目录，假设为 `/path/to/huggingface/tokenizer`。执行如下命令处理数据集：

   ```shell
   python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
     --input /path/data.json \
     --output-prefix /path/megatron_data \
     --tokenizer-type HuggingFaceTokenizer \
     --tokenizer-dir /path/to/huggingface/tokenizer
   ```

### 模型预训练

MindSpore Transformers 推荐用户使用 Megatron 数据集进行模型预训练，根据 [数据预处理](#数据预处理) 可以生成预训练数据集，下面介绍如何在动态图模式的配置文件中使用 Megatron 数据集。

修改模型配置文件中 `train_dataset` 部分内容：

```yaml
train_dataset:
  dataloader:
    type: BlendedMegatronDatasetDataLoader
    datasets_type: GPTDataset
    sizes:
      - 8000   # 训练集数据样本数
      - 0      # 测试集数据样本数，当前不支持配置
      - 0      # 评测集数据样本数，当前不支持配置
    config:
      seed: 1234                         # 数据采样随机种子
      split: "1, 0, 0"                   # 训练、测试、评测集使用比例，当前不支持配置
      seq_length: 8192                   # 数据集返回数据的序列长度
      eod_mask_loss: false               # 是否在 eod 处计算 loss
      reset_position_ids: false          # 是否在 eod 处重置 position_ids
      create_attention_mask: false       # 是否返回 attention_mask
      reset_attention_mask: false        # 是否在 eod 处重置 attention_mask，返回阶梯状 attention_mask
      create_compressed_eod_mask: false  # 是否返回压缩后的 attention_mask
      eod_pad_length: 128                # 设置压缩后 attention_mask 的长度
      eod: 1                             # 数据集中 eod 的 token id
      pad: -1                            # 数据集中 pad 的 token id
      data_path:                         # Megatron 数据集采样比例以及路径
        - "1"                            # 数据集占比
        - "/path/megatron_data"          # 数据集 bin 文件路径（去除 .bin 后缀）

    column_names: ["input_ids", "labels", "loss_mask", "position_ids"]
    shuffle: false
    python_multiprocessing: false

  drop_remainder: true
  num_parallel_workers: 8
  prefetch_size: 1
  numa_enable: false
```

下面是对 `BlendedMegatronDatasetDataLoader` 各配置项的说明：

| 参数名 | 说明 |
|--------|------|
| `datasets_type` | Megatron 数据集类型，当前仅支持 `GPTDataset` |
| `sizes` | 长度为 3 的列表，分别表示训练集、测试集、评测集的样本数，当前仅训练集有效 |
| `config.seed` | 数据集采样的随机种子，Megatron 数据集会根据该值对样本进行随机采样和拼接，默认值为 `1234` |
| `config.split` | 训练、测试、评测集使用比例，以逗号分隔，当前不支持配置 |
| `config.seq_length` | 数据集返回数据的序列长度，应该与训练模型的序列长度一致 |
| `config.eod_mask_loss` | 是否在 eod 处计算 loss，默认值为 `false` |
| `config.reset_position_ids` | 是否在 eod 处重置 position_ids，默认值为 `false` |
| `config.create_attention_mask` | 是否返回 attention_mask，默认值为 `true` |
| `config.reset_attention_mask` | 是否在 eod 处重置 attention_mask，返回阶梯状 attention_mask，仅在 `create_attention_mask=true` 时生效，默认值为 `false` |
| `config.create_compressed_eod_mask` | 是否返回压缩后的 attention_mask（即 `actual_seq_len`），优先级高于 `create_attention_mask`，默认值为 `false` |
| `config.eod_pad_length` | 设置压缩后 attention_mask 的长度，仅在 `create_compressed_eod_mask=true` 时生效，默认值为 `128` |
| `config.eod` | 数据集中 eod 的 token id |
| `config.pad` | 数据集中 pad 的 token id |
| `config.data_path` | 列表，每连续两个列表元素（数字，字符串）被视作一个数据集，分别表示该数据集的采样占比和数据集 bin 文件去掉后缀 `.bin` 的路径，所有数据集的占比之和应当为 1 |

此外，`column_names` 需要根据 `create_attention_mask` 和 `create_compressed_eod_mask` 的配置进行调整：

- 当 `create_compressed_eod_mask=true` 时：

  ```yaml
  column_names: ["input_ids", "labels", "loss_mask", "position_ids", "actual_seq_len"]
  ```

- 当 `create_compressed_eod_mask=false` 且 `create_attention_mask=true` 时：

  ```yaml
  column_names: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
  ```

- 当 `create_compressed_eod_mask=false` 且 `create_attention_mask=false` 时：

  ```yaml
  column_names: ["input_ids", "labels", "loss_mask", "position_ids"]
  ```

修改模型配置文件中数据集相关配置项之后，即可参考模型文档拉起模型预训练任务。

## HuggingFace 数据集

MindSpore Transformers 对接了 [HuggingFace 数据集](https://huggingface.co/datasets)（以下简称 HF 数据集）模块，提供了高效灵活的 HF 数据集加载与处理功能，主要特性包括：

1. **多样化数据加载**：支持 HuggingFace `datasets` 库的多种数据格式与加载方式，轻松适配不同来源与结构的数据。
2. **丰富的数据处理接口**：兼容 `datasets` 库的多种数据处理方法（如 `sort`、`flatten`、`shuffle` 等），满足常见预处理需求。
3. **可扩展的数据操作**：支持用户自定义数据集处理逻辑，并提供高效的数据 **packing 功能**，适合大规模训练场景下的优化。

> 在 MindSpore Transformers 中使用 HuggingFace 数据集需要了解 `datasets` 第三方库的数据集加载与处理等基本功能，可参考 [链接](https://huggingface.co/docs/datasets/loading) 进行查阅。
>
> 如果使用 Python 版本小于 3.10，则需要安装 aiohttp 3.8.1 以下版本。

### 配置说明

在动态图模式的模型训练任务中使用 HF 数据集功能，需要在 YAML 文件中修改 `train_dataset` 相关配置：

```yaml
train_dataset:
  dataloader:
    type: HFDataLoader

    # datasets load arguments
    load_func: 'load_dataset'
    path: "json"
    data_files: "/path/alpaca-gpt4-data.json"
    split: "train"

    # MindSpore Transformers dataset arguments
    create_attention_mask: true
    create_compressed_eod_mask: false
    compressed_eod_mask_length: 128
    shuffle: false

    # dataset process arguments
    handler:
      - type: AlpacaInstructDataHandler
        seq_length: 4096
        padding: false
        tokenizer:
          pretrained_model_dir: '/path/qwen3'
          trust_remote_code: true
          padding_side: 'right'
      - type: PackingHandler
        seq_length: 4096
        pack_strategy: 'pack'

    column_names: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
    python_multiprocessing: false

  drop_remainder: true
  num_parallel_workers: 8
  prefetch_size: 1
  numa_enable: false
```

> 所有示例中涉及的 `seq_length`、`tokenizer` 等参数均来自 `qwen3` 模型。

`dataloader` 中参数说明：

| 参数名 | 描述 | 类型 |
|--------|------|:----:|
| `type` | 固定为 `HFDataLoader`，该模块支持 HuggingFace 开源社区的数据集加载与处理功能 | str |
| `load_func` | 指定加载数据集调用接口，可选值为 `load_dataset` 和 `load_from_disk`，具体配置说明见 [数据集加载](#数据集加载)，默认值为 `load_dataset` | str |
| `create_attention_mask` | 是否在数据集迭代过程中返回对应的 attention mask，默认值为 `false` | bool |
| `create_compressed_eod_mask` | 是否在数据集迭代过程中返回经过压缩的一维 attention mask（即 `actual_seq_len`），默认值为 `false` | bool |
| `compressed_eod_mask_length` | 生成压缩 attention mask 的长度，通常为数据集内各样本中 eod token 个数的最大值，默认值为 `128` | int |
| `shuffle` | 是否对数据集进行随机采样，默认值为 `false` | bool |
| `handler` | 数据预处理操作，具体介绍可参考 [数据集处理](#数据集处理) 章节 | list |

### 数据集加载

数据集加载功能主要通过 `load_func` 参数实现。`HFDataLoader` 会将 [配置说明](#配置说明) 中之外的所有参数作为数据集加载接口的入参，具体使用说明如下：

1. 使用 `datasets.load_dataset` 接口加载数据集：

   在数据集配置中设置 `load_func: 'load_dataset'`，同时配置如下参数：

   1. **path (str)** — 数据集文件夹的路径或名称

    - 如果 path 是本地目录，则从该目录中的支持文件（csv、json、parquet 等）加载数据集，例如：`'/path/json/'`；
    - 如果 path 是某个数据集构建器的名称，并且指定了 data_files 或 data_dir（可用的构建器包括 "json", "csv", "parquet", "arrow" 等），则从 data_files 或 data_dir 中的文件加载数据集。

   2. **data_dir (str, 可选)** — 当 path 配置为数据集构建器的名称时，指定数据集文件夹路径。

   3. **data_files (str, 可选)** — 当 path 配置为数据集构建器的名称时，指定数据集文件路径，可以是单个文件或包含多个文件路径的列表。

   4. **split (str)** — 要加载的数据切分。如果为 None，将返回包含所有切分的字典（通常是 datasets.Split.TRAIN 和 datasets.Split.TEST）；如果指定，则返回对应切分的 Dataset 实例。

2. 使用 `datasets.load_from_disk` 接口加载数据集：

   在数据集配置中设置 `load_func: 'load_from_disk'`，同时配置如下参数：

   - **dataset_path (str)** — 数据集文件夹路径，通常使用该接口加载使用 `datasets.save_to_disk` 保存的数据集。

### 数据集流式加载

在使用样本数非常多的数据集时，可能会存在设备内存不足的问题，可以通过使用流式加载来降低内存负载，该功能原理及相关说明可参考 [stream](https://huggingface.co/docs/datasets/v4.0.0/en/stream)。

开启数据集流式加载功能需要在 [配置说明](#配置说明) 中 `dataloader` 中添加如下配置：

```yaml
train_dataset:
  dataloader:
    type: HFDataLoader
    streaming: true
    size: 2000
    dataset_state_dir: '/path/dataset_state_dir'
    # ... 其他配置
```

参数说明：

| 参数名 | 描述 | 类型 |
|--------|------|:----:|
| `streaming` | 是否开启数据集流式加载功能 | bool |
| `size` | 指定数据集迭代总样本数，以流式模式加载数据集将创建一个 [IterableDataset](https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.IterableDataset) 实例，在迭代所有数据的前提下无法获取总样本数，因此需要指定该参数 | int |
| `dataset_state_dir` | 指定保存和加载数据集状态文件夹，主要用于在保存权重时同步保存数据集状态以及加载进行断点续训。<br/>由于 MindSpore 数据集默认开启数据下沉功能，数据集状态会在权重保存之前进行保存；<br/>在使用流式加载数据集进行断点续训时，修改影响 `global_batch_size` 的参数（如 `data_parallel`、`batch_size` 等），会导致无法续训并重新采样进行训练 | str |

目前流式加载功能在以下预处理场景经过验证：

1. Alpaca 数据集预处理，相关配置：`AlpacaInstructDataHandler`；
2. Packing 数据集预处理，相关配置：`PackingHandler`；
3. 重命名列操作，相关配置：`rename_column`；
4. 移除列操作，相关配置：`remove_columns`。

### 数据集处理

`HFDataLoader` 支持 datasets 原生数据处理以及用户自定义处理操作，数据预处理操作主要通过 `handler` 机制实现，该模块会按照配置顺序执行数据预处理操作。

#### 原生数据处理功能

如果要实现重命名数据列、移除数据列、随机采样数据集功能，可进行如下配置：

```yaml
handler:
  - type: 'rename_column'
    original_column_name: 'col1'
    new_column_name: 'col2'
  - type: 'remove_columns'
    column_names: 'col2'
  - type: 'shuffle'
    seed: 42
```

1. rename_column - 重命名数据列

   示例中配置可以将 `col1` 重命名为 `col2`。

2. remove_columns - 移除数据列

   示例中配置可以将重命名后的 `col2` 移除。

3. shuffle - 随机打乱数据集

   示例中配置以 42 为随机种子，对数据集进行随机采样。

其他 datasets 原生数据处理可参考 [datasets process](https://huggingface.co/docs/datasets/process) 文档。

#### 自定义数据处理功能

自定义数据预处理功能需要用户自己实现数据处理模块。下面介绍自定义数据处理模块实现过程，可参考 [AlpacaInstructDataHandler](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/dataset/handler/alpaca_handler.py)。

用户自定义数据处理支持 `Class` 和 `Method` 两种形式：

如果使用 `Class` 构造数据处理模块：

1. 实现包含 `__call__` 函数的 `Class`

   ```python
   class CustomHandler:
       def __init__(self, seed):
           self.seed = seed

       def __call__(self, dataset):
           dataset = dataset.shuffle(seed=self.seed)
           return dataset
   ```

   上面的 `CustomHandler` 实现了数据集随机采样的处理操作。如果要实现其他功能，可以修改数据预处理操作并返回处理后的数据集。

   同时，MindSpore Transformers 提供了 [BaseInstructDataHandler](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/dataset/handler/base_handler.py)，并内置了 tokenizer 配置功能。如果需要使用 tokenizer，可以继承 `BaseInstructDataHandler` 类。

2. 在 [\_\_init\_\_.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/dataset/handler/__init__.py) 中添加调用

   ```python
   from .custom_handler import CustomHandler
   ```

3. 在配置中使用 `CustomHandler`

   ```yaml
   handler:
     - type: CustomHandler
       seed: 42
   ```

如果使用 `Method` 构造数据处理模块：

1. 实现包含 dataset 实例入参的函数

   ```python
   def custom_process(dataset, seed):
       dataset = dataset.shuffle(seed)
       return dataset
   ```

2. 在 [\_\_init\_\_.py](https://atomgit.com/mindspore/mindformers/blob/master/mindformers/dataset/handler/__init__.py) 中添加调用

   ```python
   from .custom_handler import custom_process
   ```

3. 在配置中使用 `custom_process`

   ```yaml
   handler:
     - type: custom_process
       seed: 42
   ```

### 应用实践

下面以 `qwen3` 模型以及 `alpaca` 数据集为例，介绍如何使用 HF 数据集进行微调。需要使用 `AlpacaInstructDataHandler` 对数据进行在线处理，具体参数说明如下。

- `seq_length`：通过 tokenizer 将文本编码为 token id 的最大长度，通常与模型训练的序列长度一致。
- `padding`：是否在 tokenizer 编码时将 token id 填充到最大长度。
- `tokenizer`：`pretrained_model_dir` 表示从 HF 社区上下载的模型词表及权重文件夹，`trust_remote_code` 通常设置为 `true`，`padding_side` 表示从 token id 右侧进行填充。

#### alpaca 数据集微调

以 `qwen3` 模型微调为例，修改 `qwen3` 模型训练配置文件：

```yaml
train_dataset:
  dataloader:
    type: HFDataLoader

    # datasets load arguments
    load_func: 'load_dataset'
    path: 'json'
    data_files: '/path/alpaca-gpt4-data.json'

    # MindSpore Transformers dataset arguments
    shuffle: false

    # dataset process arguments
    handler:
      - type: AlpacaInstructDataHandler
        seq_length: 4096
        padding: true
        tokenizer:
          pretrained_model_dir: '/path/qwen3'  # qwen3 repo dir
          trust_remote_code: true
          padding_side: 'right'

    column_names: ["input_ids", "labels"]
    python_multiprocessing: false

  drop_remainder: true
  num_parallel_workers: 8
  prefetch_size: 1
  numa_enable: false
```

修改配置文件后，即可参考 `qwen3` 模型文档拉起微调任务。

#### alpaca 数据集 packing 微调

MindSpore Transformers 实现了数据集的 packing 功能，主要用于大模型训练任务中将多个短序列拼接成定长的长序列，以提升训练效率。它目前支持两种策略，可以通过 `pack_strategy` 进行配置：

1. **pack**：将多个样本拼接成一个定长序列。当待拼接样本超过最大长度 `seq_length` 后，将该样本放入下一个拼接样本中。
2. **truncate**：将多个样本拼接成一个定长序列。当待拼接样本超过最大长度 `seq_length` 后，对样本进行截断，并将剩余部分放入下一个拼接样本中。

该功能通过 `PackingHandler` 类实现，最终输出只包含 `input_ids`、`labels` 和 `actual_seq_len` 三个字段。

以 `qwen3` 模型微调为例，修改 `qwen3` 模型训练配置文件：

```yaml
train_dataset:
  dataloader:
    type: HFDataLoader

    # datasets load arguments
    load_func: 'load_dataset'
    path: 'json'
    data_files: '/path/alpaca-gpt4-data.json'

    # MindSpore Transformers dataset arguments
    shuffle: false

    # dataset process arguments
    handler:
      - type: AlpacaInstructDataHandler
        seq_length: 4096
        padding: false
        tokenizer:
          pretrained_model_dir: '/path/qwen3'  # qwen3 repo dir
          trust_remote_code: true
          padding_side: 'right'
      - type: PackingHandler
        seq_length: 4096
        pack_strategy: 'pack'

    column_names: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
    python_multiprocessing: false

  drop_remainder: true
  num_parallel_workers: 8
  prefetch_size: 1
  numa_enable: false
```

修改配置文件后，即可参考 `qwen3` 模型文档拉起微调任务。

## MindRecord 数据集

MindRecord 是 MindSpore 提供的高效数据存储/读取模块，可以减少磁盘 IO、网络 IO 开销，从而获得更好的数据加载体验，更多具体功能介绍可参考 [文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mindrecord.html)，这里仅对如何在 MindSpore Transformers 动态图模式训练任务中使用 MindRecord 进行介绍。

下面以 `qwen3-8b` 进行微调为示例进行相关功能说明，示例中的脚本仅适用于指定数据集，如果需要对自定义数据集进行处理，可以参考 [MindRecord 格式转换](https://www.mindspore.cn/tutorials/zh-CN/master/dataset/record.html) 进行数据预处理。

### 数据预处理

1. 下载 `alpaca` 数据集：[链接](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

2. 执行数据处理脚本 [alpaca_converter.py](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/qwen3/alpaca_converter.py) 将 `alpaca` 数据集转换为对话形式：

   ```shell
   python alpaca_converter.py \
     --data_path /path/alpaca_data.json \
     --output_path /path/alpaca-data-messages.json
   ```

   其中，`data_path` 表示下载后 `alpaca` 数据集的路径，`output_path` 表示生成对话形式数据文件的保存路径。

3. 执行脚本 [datasets_preprocess.py](https://atomgit.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/example/qwen3/datasets_preprocess.py) 将对话形式的数据文件转换为 MindRecord 格式：

   ```shell
   python datasets_preprocess.py \
     --input_glob /path/alpaca-data-messages.json \
     --tokenizer_dir /path/Qwen3-8B \
     --seq_length 32768 \
     --output_file /path/alpaca-messages.mindrecord
   ```

   该脚本各参数说明如下：

   - `input_glob`：生成对话形式数据文件路径
   - `tokenizer_dir`：qwen3 的文件路径
   - `seq_length`：生成 MindRecord 数据的序列长度
   - `output_file`：生成 MindRecord 数据的保存路径

### 模型微调

参考上述数据预处理流程可生成用于 `qwen3-8b` 模型微调的 MindRecord 数据集，下面介绍如何使用生成的数据文件启动模型微调任务。

1. 修改模型配置文件

   `qwen3-8b` 模型微调使用 `finetune_qwen3.yaml` 配置文件，修改其中数据集部分配置：

   ```yaml
   train_dataset:
     dataloader:
       type: MindDataset
       dataset_files: "/path/alpaca-messages.mindrecord"
       shuffle: true

     drop_remainder: true
     num_parallel_workers: 8
     prefetch_size: 1
     numa_enable: false
   ```

   在模型训练任务中使用 MindRecord 数据集需要修改 `dataloader` 中的配置项：

   - `type`：data_loader 类型，使用 MindRecord 数据集设置为 `MindDataset`
   - `dataset_files`：MindRecord 数据文件路径，可以是单个 `.mindrecord` 文件路径、包含多个文件路径的列表，或包含 `.mindrecord` 文件的目录路径
   - `shuffle`：是否在训练时对数据样本进行随机采样

2. 启动模型微调

   修改模型配置文件中数据集相关配置项之后，即可参考模型文档拉起模型微调任务，这里以 [Qwen3 模型文档](https://atomgit.com/mindspore/mindformers/blob/master/configs/qwen3/README.md) 为例。

### 多源数据集

MindSpore 框架原生数据集加载模块 [MindDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.MindDataset.html)，在对多个 MindRecord 数据集进行加载和采样时存在性能等瓶颈，因此 MindSpore Transformers 通过 `MultiSourceDataLoader` 实现多个数据集高效加载与采样功能。

多源数据集功能主要通过修改配置文件中 `dataloader` 配置开启，下面提供示例：

```yaml
train_dataset:
  dataloader:
    type: MultiSourceDataLoader
    data_source_type: random_access
    shuffle: true
    dataset_ratios: [0.2, 0.8]
    samples_count: 1000
    nums_per_dataset: [2000, 2000]
    sub_data_loader_args:
      stage: 'train'
      column_names: ["input_ids", "target_ids", "attention_mask"]
    sub_data_loader:
      - type: MindDataset
        dataset_files: "/path/alpaca-messages.mindrecord"
      - type: MindDataset
        dataset_files: "/path/alpaca-messages.mindrecord"
    load_indices_npz_path: '/path/index.npz'
    save_indices_npz_path: '/path/index.npz'

  drop_remainder: true
  num_parallel_workers: 8
  prefetch_size: 1
  numa_enable: false
```

其中 `shuffle` 配置会影响 `shuffle_dataset` 和 `shuffle_file` 两个参数：

- `shuffle_dataset` 表示子数据集层面的随机采样
- `shuffle_file` 表示样本层面的随机采样

在 `shuffle` 配置不同值时，会有如下结果：

| shuffle | shuffle_dataset | shuffle_file |
|---------|:---------------:|:------------:|
| true | true | true |
| false | false | false |
| infile | false | true |
| files | true | false |
| global | true | true |

其他配置项说明如下：

| 参数名 | 说明 | 类型 |
|--------|------|:----:|
| `dataset_ratios` | 每个子数据集的采样比例，各子数据集采样比例和为 1 | list |
| `samples_count` | 每个子数据集参与采样的样本数量，仅在配置 `dataset_ratios` 时生效 | int |
| `nums_per_dataset` | 每个子数据集的样本采样数量，在不配置 `dataset_ratios` 时生效 | list |
| `sub_data_loader_args` | 每个子数据集的通用配置，在所有子数据集构建时生效 | dict |
| `sub_data_loader` | 每个子数据集的配置，与单个 MindRecord 数据集中 `dataloader` 配置相同 | list |
| `load_indices_npz_path` | 加载数据索引文件路径 | str |
| `save_indices_npz_path` | 数据索引文件保存路径 | str |
