# 数据集

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindformers/docs/source_zh_cn/function/dataset.md)

目前MindSpore Transformers的预训练和微调支持多种格式的数据集加载能力，包括Megatron多源数据集、MindRecord数据集以及HuggingFace数据集的加载方式。每种格式的数据集的具体使用方法的参考如下。

## Megatron多源数据集

Megatron多源数据集是指从多个不同来源收集的数据集，这些数据集可以包含不同的文本类型、格式和领域。使用多源数据集可以帮助模型学习到更广泛的语言特征和知识，从而提高模型的泛化能力和性能。Megatron框架目前实现的多源数据集，需要先将原数据集预处理成BIN格式的数据集。当前MindSpore Transformers已经原生适配了Megatron多源数据集，提供了制作BIN格式数据集的脚本，支持在训练任务中直接使用Megatron多源数据集。

### 制作 BIN 格式数据集

MindSpore Transformers 提供了一个预处理脚本 [mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py) 将文本数据转换成BIN格式数据集，该脚本当前仅支持处理特定形式的 JSON 格式的文件。用户需要先将原始数据集文件转换成特定形式的JSON格式的文件，再使用预处理脚本生成BIN格式的数据集文件。当前 MindSpore Transformers 中的一些模型已经提供了将特定开源数据集转换成特定形式 JSON 格式文件的脚本，用户如想使用自有数据集，则需要通过自行编写脚本的方式将其转换为所需形式。

所需的 JSON 格式文件内容的形式如下：

```json
{"id": "0", "text": "The quick brown fox", "type": "Eng", "src": "www.nvidia.com", "title": "First Part"}
{"id": "1", "text": "jumps over the lazy dog", "type": "Eng", "src": "The Internet", "title": "Second Part"}
...
```

其中每条数据由若干键值对组成，支持的键及说明如下：

- `"id"`: 数据的编号，按顺序编号即可，必须存在
- `"text"`: 实际用作训练的文本数据，必须存在
- `"type"`: 注明语言类型，可选
- `"src"`：注明数据的来源，可选
- `"title"`：注明数据的标题，可选

下面以处理 Wiki 数据集并用作 Llama2 模型预训练为例，说明制作 BIN 格式数据集的详细步骤：

1. 下载 Wiki 数据集

   原始 Wiki 数据集的下载参考 [Llama2 数据集下载](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)。

2. 生成 JSON 格式文件

   Wiki 数据集的原始格式如下：

   ![](image/wikitext_sample.png)

   将 Wiki 数据集处理后的 JSON 文件 `wiki.json` 的格式如下 （省略长文本）：

   ```json
   {"id": 0, "text": "The gold dollar or gold one ..."}
   {"id": 1, "text": "Super Mario Land is a 1989 ..."}
   {"id": 2, "text": "The Sinclair Scientific Programmable ..."}
   ...
   ```

3. 下载 Llama2 的词表文件

   预处理脚本中会把原始文本数据使用模型的分词器 Tokenizer 处理成 Tokens 的形式，因此需要提前下载词表文件。

   Llama2 词表文件的下载链接：[tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

4. 使用预处理脚本生成 BIN 格式文件

    处理成上述这样特定的 JSON 格式的文件后，调用 [mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py) 将其转换成BIN格式的数据集，具体命令如下：

    ```shell
    python mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py \
    --input ./wiki.json \
    --output-prefix wiki_processed_1024 \
    --tokenizer-type LlamaTokenizer \
    --vocab-file ./tokenizer.model \
    --add_bos_token True \
    --add_eos_token True \
    --pad_or_stitch stitch \
    --seq-length 1024 \
    --workers 1
    ```

    配置参数说明：

    - `--input`: JSON 格式文件的路径
    - `--output-prefix`: 预处理后的输出文件的文件名前缀
    - `--tokenizer-type`: 模型对应的 tokenizer 的类型
    - `--vocab-file`: 模型分词器 Tokenizer 的词表文件路径
    - `--add_bos_token`: 是否在数据的首位置添加 bos_token，默认为 False
    - `--add_eos_token`: 是否在数据的末位置添加 eos_token，默认为 False
    - `--pad_or_stitch`: 根据训练任务的要求，设置是否拼接还是补齐，pad为补齐模式，该模式会将长度不足的数据补齐至seq-length长度；stitch为拼接模式，该模式会将多条数据拼接成长度为seq-length的数据
    - `--seq-length`: 预处理后每条数据长度
    - `--workers`: 预处理时并行 worker 的数量

执行以上命令之后，会得到两个文件，分别为 `.bin` 和 `.idx` 格式的文件，其中 `.bin` 格式文件存储实际的数据，`.idx` 格式文件存储每条数据的索引。

### 在训练任务中使用多源数据集

按照如下方式在训练任务中使用Megatron多源数据集：

1. 准备`parallel_speed_up.json`文件

   `parallel_speed_up.json` 是数据集并行通信配置文件，文件内容如下：

   ```json
   {
       "dataset_broadcast_opt_level": 3
   }
   ```

2. 设置环境变量

    在命令行输入如下命令设置环境变量：

    ```shell
    export MS_DEV_DYNAMIC_SINK1=False
    ```

3. 修改训练任务的 YAML 配置文件

    在 YAML 配置文件中配置Megatron多源数据集的相关参数。此处，以 Llama2-7B 模型预训练任务来举例说明，修改 [pretrain_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/pretrain_llama2_7b.yaml#L39) 中的 `train_dataset` 、 `runner_config` 、 `parallel_config` 、 `parallel` 以及 `context` 配置项。具体修改及说明如下：

    ```yaml
    train_dataset: &train_dataset
      data_loader:
        type: BlendedMegatronDatasetDataLoader
        datasets_type: "GPTDataset"
        sizes:
          - 1000
          - 0
          - 0
        shuffle: False
        config:
          seed: 1234
          seq_length: 1024
          split: "1, 0, 0"
          data_path:
            - 0.3
            - "/path/to/my_wiki_test_1024_text_document"
            - 0.7
            - "/path/to/my_wiki_test_1024_text_document"
          num_dataset_builder_threads: 1
          eod_mask_loss: False
          create_attention_mask: False
      input_columns: ["input_ids", "labels", "loss_mask", "position_ids"]
    ```

    其中：

    - data_loader.type：dataloader 的类型，需设置为 `BlendedMegatronDatasetDataLoader` 。
    - data_loader.datasets_type：数据集类型，当前仅支持 `GPTDataset` 。
    - data_loader.sizes：`- 1000` ， `- 0` ， `- 0` 分别为训练集、测试集以及验证集采样的大小，当前只支持配置训练集。
    - input_columns：设置训练数据集输入的数据列，一般配置为 `["input_ids", "labels", "loss_mask", "position_ids"]` 。
    - data_loader.config.seed: 创建数据集时的随机数种子，默认值： `1234` 。
    - data_loader.config.seq_length: 每条数据的长度，必须和 YAML 配置中的 model.model_config.seq_length 保持一致。
    - data_loader.config.split：分割字符串，用逗号分隔训练集、测试集以及验证集的比重，用于从单个分布中绘制样本时分割数据集，当前只支持配置为 `"1, 0, 0"` 。
    - data_loader.config.data_path：数字是每个数据集的比重，字符串是数据集 BIN 文件的路径，路径需要去掉文件格式后缀 `.bin` 。
    - data_loader.config.num_dataset_builder_threads：创建数据集时使用的进程数，默认值： `1` 。
    - data_loader.config.eod_mask_loss：是否使用 eod mask 的开关，默认值： `False` 。
    - data_loader.config.create_attention_mask：是否构造 attention_mask，默认值：`True` 。

    当前多源数据集目前还存在限制，仅支持非 full batch 的场景，且不支持序列流水线并行特性，需要根据以下对相应配置项进行修改：

    ```yaml
    runner_config:
        sink_mode: True
        sink_size: 1

    parallel_config:
        data_parallel: &dp 2
        model_parallel: 2
        pipeline_stage: 1

    parallel:
        full_batch: False
        dataset_strategy: [[*dp, 1], [*dp, 1], [*dp, 1], [*dp, 1]]

    context:
        ascend_config:
            parallel_speed_up_json_path: "/path/to/parallel_speed_up.json"
    ```

    需要注意的配置说明如下：

    - parallel.dataset_strategy：仅支持 List of List 类型，List中子List的个数需要等于 train_dataset.input_columns 的长度，并且 List 中的每个子 List 需要和数据集返回的数据的shape保持一致。一般在数据的第1维进行数据并行切分，所以子List的第1位数配置成 `*dp` ，其他位配置为 `1` 。具体原理可以参考[数据集切分](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/parallel/dataset_slice.html)。

4. 编译 Megatron 数据集模块

    MindSpore Transformers 内置了 Megatron 的数据集模块代码，需要在启动训练任务之前执行如下命令进行编译：

    ```shell
    pip install pybind11
    cd mindformers/dataset/blended_datasets
    make
    ```

## MindRecord 数据集

MindRecord 是由 MindSpore 开发的一种高效数据格式，用于存储机器学习或深度学习的数据集。

MindRecord 格式旨在提高数据处理效率，尤其是在大规模数据训练场景下，可以更快地加载和处理数据。
MindRecord 文件通常包含了模型训练所需的输入样本，这些样本经过预处理（如编码、归一化等），以优化读取速度和内存使用。

更多关于 MindRecord 相关接口的实现及案例，请参考 [MindSpore 中关于 《MindRecord》 的相关文档](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/api_python/mindspore.mindrecord.html)

### 如何制作 MindRecord 数据集

MindRecord 模块提供了一些方法帮助用户将不同数据集转换为 MindRecord 格式，
用户可以使用由 MindSpore 提供的 FileWriter 接口生成 MindRecord 格式数据集。

下面将以 Llama2 为例，提供一个基于 json 格式文件制作 MindRecord 数据集的案例：

1. 准备 json 文件；

   准备类似这样的 json 文件，命名为 `mydata.json` ：

   ```json
   [
      {
        "text": "I love Beijing, because it is a city that beautifully blends rich history with modern vibrancy."
      },
      {
        "text": "I love Hangzhou, because it is a city that seamlessly combines natural beauty with rich cultural heritage."
      }
   ]
   ```

2. 读取 json 文件；

   ```python
   import json

   raw_data = None
   file = open("mydata.json", "r")  # 打开 json 文件
   if file is not None:
      raw_data = json.load(file)  # 读取 json 文件到 raw_data 中
      file.close()
   ```

3. 定义一个 MindRecord 的 ``schema`` ，并创建一个 ``FileWriter`` 对象；

    ```python
    from mindspore.mindrecord import FileWriter

    # 定义一个 MindRecord 的 schema
    schema = {'input_ids': {"type": "int32", "shape": [-1]}}
    # 创建一个 FileWriter 对象
    writer = FileWriter(file_name="output_file", shard_num=1)
    writer.add_schema(schema, "dataset_type")
    ```

4. 遍历处理 json 文件中的每一条数据，将其转换为 MindRecord 格式，并写入 MindRecord 文件中。

   词表下载链接： [tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

    ```python
    import numpy as np
    from mindformers import LlamaTokenizer

    def tokenize_json(tokenizer, raw_data):
        """tokenize json file dataset"""
        content = [] # 读取每个 json 数据，获取其 "input_ids"
        for line in raw_data:
            stripped_line = line['text'].strip()
            if stripped_line:
                line_ids = tokenizer(stripped_line)["input_ids"]
                content.append(line_ids)

        for ids in content:
            sample = {}
            sample['input_ids'] = np.array(ids, dtype=np.int32)
            yield sample

    # 将文本数据分词
    word_tokenizer = LlamaTokenizer(vocab_file=r"tokenizer.model")

    # 遍历处理 json 文件中的每一条数据，将其转化为 MindRecord 格式后写入 MindRecord 文件
    # tokenize_json 为自定义的对 json 中数据进行分词的方法
    for x in tokenize_json(word_tokenizer, raw_data):
        writer.write_raw_data([x])
    writer.commit()
    ```

详细案例可以参考 [Llama2 中的数据预处理案例](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)。

### 在任务中使用 MindRecord 格式数据集

通过在 yaml 配置文件中配置数据集相关参数，可以让训练或评测任务使用准备好的 MindRecord 格式数据集。

此处，以 Llama2-7B 模型预训练任务来举例说明，在 [pretrain_llama2_7b.yaml 文件](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/pretrain_llama2_7b.yaml#L39) 中的默认配置参数及说明如下：

```yaml
# dataset
train_dataset: &train_dataset
  data_loader:
    type: MindDataset
    dataset_dir: ""
    shuffle: True
  input_columns: ["input_ids"]
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 6
  repeat: 1
  numa_enable: False
  prefetch_size: 1

train_dataset_task:
  type: CausalLanguageModelDataset
  dataset_config: *train_dataset
```

配置如下参数以使用 MindRecord 格式数据集：

- data_loader.type：dataloader 的类型，此处需要设置为 `MindDataset` 。
- data_loader.dataset_dir：数据集文件路径。
- input_columns：设置训练数据集输入的数据列。当前为预训练场景，设置为 `["input_ids"]` 。

其余参数介绍可以参考 [配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/appendix/conf_files.html) 的 “模型训练配置” 和 “模型评估配置”。

## HuggingFace数据集

目前数据集加载功能已接入 [魔乐开源社区](https://modelers.cn/datasets)、[HuggingFace社区](https://huggingface.co/datasets)，并支持数据集在线加载与预处理，同时还可对数据集进行[packing](#数据集packing)，提升模型训练效率。

### 使用说明

HuggingFace数据集可实现HuggingFace社区以及魔乐开源社区中的数据集在线、离线加载，下面主要针对环境准备、数据集加载流程、以及在如何在配置文件中配置使用HuggingFace数据集功能进行介绍。

#### 对接开源社区

- 对接HuggingFace社区

   如果需要使用HuggingFace社区中的数据集需要执行如下步骤：

  1. 环境准备

     环境变量 `HF_ENDPOINT` 可以控制开源社区huggingFace实际使用的远程仓库，未配置时默认为 `https://huggingFace.co` ，
     针对国内环境，需要配置成镜像地址 ```export HF_ENDPOINT=https://hf-mirror.com``` 。

  2. 安装依赖

     ```shell
     pip install datasets
     ```

- 对接魔乐开源社区

   如果需要使用魔乐开源社区中的数据集需要执行如下步骤：

  1. 环境准备

     环境变量 `OPENMIND_HUB_ENDPOINT` 可以控制魔乐开源社区实际使用的远程仓库，
     未配置时默认为 ```export OPENMIND_HUB_ENDPOINT=https://telecom.openmind.cn``` 。

  2. 安装依赖

     ```shell
     git clone https://gitee.com/openmind-ai/openmind-hub.git
     cd openmind-hub
     pip install -e .
     cd ..
     git clone https://gitee.com/foundation-models/openmind-datasets.git
     cd openmind-datasets
     pip install -e .
     cd ..
     ```

> 当环境安装了 openmind-datasets 三方件时，默认对接的是魔乐开源社区，如果这是想对接 HuggingFace，环境变量 `USE_OM` 可以控制具体对接哪个社区，默认值为 `ON` 为魔乐社区，修改为 `OFF` 对接 HuggingFace 社区

#### 数据集加载流程

![commondataloader.png](image/commondataloader.png)

在线数据集加载与处理功能主要通过`CommonDataLoader`实现，其中数据加载部分可通过配置文件进行自定义配置，具体配置内容可参考[dataloader参数说明](#dataloader参数说明)，在线加载模块需要用户针对不同数据集进行自定义实现，如通过`AlpacaInstructDataHandler`类可实现对`alpaca`数据集进行预处理，具体实现过程可参考[自定义数据handler](#自定义数据handler)。

#### dataloader参数说明

在线数据集加载功能通过在配置文件中对`data_loader`进行配置来使能，下面是在线数据集加载相关配置的示例：

```yaml
train_dataset: &train_dataset
  input_columns: &input_columns ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
  construct_args_key: *input_columns
  data_loader:
    type: CommonDataLoader
    load_func: 'load_dataset'
    shuffle: False
    split: "train"
    path: "llm-wizard/alpaca-gpt4-data"
    packing: pack
    handler:
      - type: AlpacaInstructDataHandler
        tokenizer_name: llama2_7b
        seq_length: 4096
        prompt_key: "conversations"
        output_columns: ["input_ids", "labels"]
        is_dynamic: False
      - type: PackingHandler
        seq_length: 4096
        output_columns: ["input_ids", "labels", "actual_seq_len"]
    adaptor_config:
      compress_mask: False
    column_names: *input_columns
```

其中`data_loader`中相关参数说明如下：

| 参数名            | 概述                                                                                                                                                   |  类型  |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|:----:|
| type           | 固定为`CommonDataLoader`，该模块支持HuggingFace以及魔乐开源社区的数据集加载功能                                                                                               | str  |
| packing        | 使用`handler`处理数据集时packing配置项，可选值为`pack`或`truncate`                                                                                                    | str  |
| load_func      | 加载数据集调用接口名，可选值为`load_dataset`或`load_from_disk`，读取通过`save_to_disk`接口保存的数据使用`load_from_disk`，其他场景使用`load_dataset`，默认值为`load_dataset`                   | str  |
| path           | 在`load_func=load_dataset`时，该参数含义与[datasets.load_dataset](https://huggingface.co/docs/datasets/loading)中接口相同，在`load_func=load_from_disk`时，该参数为加载数据集路径 | str  |
| data_files     | 在`load_func=load_dataset`时，该参数含义与[datasets.load_dataset](https://huggingface.co/docs/datasets/loading)中接口相同，在`load_func=load_from_disk`时不生效          | str  |
| handler        | 可配置多个`handler`，按配置顺序对加载后的数据集进行预处理，`handler`配置说明参考[自定义数据handler](#自定义数据handler)中的handler参数说明                                                          | list |
| adaptor_config | 在模型训练过程中数据集的相关配置，当前支持设置`compress_mask`，在设置`packing`时生效，开启后返回压缩后的数据掩码，默认为`False`                                                                      | dict |
| shuffle        | 是否在读取数据集时开启随机采样                                                                                                                                      | bool |
| column_names   | 设置数据集返回的列名，不指定时返回所有列                                                                                                                                 | list |
| is_dynamic     | 设置数据集返回动态长度的数据，默认为`False`                                                                                                                            | bool |

> 除了以上配置外，[datasets.load_dataset](https://huggingface.co/docs/datasets/loading)接口中的所有配置均已支持，且参数含义与功能相同。

数据集在配置packing之后返回`actual_seq_len`数据列，其含义可参考[文档](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0027.html)中`actual_seq_qlen`以及`ctual_seq_kvlen`参数介绍。

### 功能介绍

#### 动态序列长度微调

`CommonDataLoader`支持加载HuggingFace数据集进行动态shape微调，HuggingFace数据集加载分为在线加载和离线加载，下面以`alpaca`数据集为例介绍如何配置动态shape微调。

- 在线加载

  在线数据名称为`llm-wizard/alpaca-gpt4-data`，可在[HuggingFace官网](https://huggingface.co/datasets)搜索名称进行下载或使用在线名称进行加载；

  在线加载配置文件示例：

  ```yaml
  train_dataset: &train_dataset
    input_columns: &input_columns ["input_ids", "labels"]
    dynamic_batch: True                    # 开启动态shape
    divisor: 32                            # 配置divisor和remainder后，动态shape中seq_length会成为divisor的倍数以及remainder的和
    remainder: 1
    data_loader:
      type: CommonDataLoader
      shuffle: True
      split: "train"                       # 在线数据集子集名称
      path: "llm-wizard/alpaca-gpt4-data"  # 在线数据集名称
      handler:
        - type: AlpacaInstructDataHandler
          tokenizer_name: llama2_7b
          seq_length: 4096
          prompt_key: "conversations"
          output_columns: *input_columns
          is_dynamic: True
    seed: 0
    num_parallel_workers: 8
    python_multiprocessing: False
    drop_remainder: True
    repeat: 1
    numa_enable: False
    prefetch_size: 1
  ```

   1. `train_dataset`中参数说明可参考[文档](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/appendix/conf_files.html)；

   2. `AlpacaInstructDataHandler`是针对`alpaca`数据集开发的在线处理脚本，如果使用其他数据集，用户需要参考[自定义数据handler](#自定义数据handler)完成自定义数据处理的功能实现。

- 离线加载

  离线加载需要准备好`alpaca`数据集中的json文件，离线配置与在线配置仅如下配置项不同。

  ```yaml
   train_dataset:
     data_loader:
       path: "json"                               # load_dataset接口加载文件格式
       data_files: '/path/alpaca_gpt4_data.json'  # alpaca数据集文件路径
   ```

配置完数据集加载方式之后，还需要在模型配置中修改`is_dynamic=True`来开启模型动态shape训练。

```yaml
model_config:
  is_dynamic: True
```

由于动态shape会存在算子编译缓存，当运行环境内存有限时，推荐配置如下环境变量来限制编译缓存的数量，避免出现内存不足的问题：

```shell
export ACLNN_CACHE_LIMIT=10
export MS_DEV_RUNTIME_CONF="aclnn_cache_queue_length:64"
```

- `ACLNN_CACHE_LIMIT`参数说明参考[文档](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/envvar/envref_07_0031.html)。
- `MS_DEV_RUNTIME_CONF`是MindSpore中设置算子缓存序列长度的参数，其中64代表该序列的长度，默认为1024，可根据实际环境进行调整，数值设置过小可能会影响模型训练性能。

完成以上所有配置后，即可参考具体使用的模型文档进行动态shape微调。

#### 自定义数据handler

用户可以使用自定义数据 handler 逻辑，对加载到的数据集进行各种数据预处理定制逻辑。

- handler参数说明

| 参数名            | 概述                                                                      |    类型    |
|----------------|-------------------------------------------------------------------------|:--------:|
| type           | 自定义数据 handler 名称，自定义handler必须继承`BaseInstructDataHandler`                |   str    |
| tokenizer_name | 使用的 tokenizer 分词器名称                                                     |   str    |
| tokenizer      | tokenizer 相关配置参数, 可以是字典或者字符串，也可以直接配置`tokenizer`对象，优先级低于`tokenizer_name` | dict/str |
| seq_length     | 处理序列的最大长度，通常与模型的序列长度相同                                                  |   int    |
| output_columns | 数据预处理后返回的数据列名                                                           |   list   |
| prompt_key     | 增加 prompt 处理后数据的列名                                                      |   str    |

- 开发样例一

自定义数据 handler 一般放在 `mindformers/dataset/handler` 目录下，自定义的需要继承抽象基类 ``BaseInstructDataHandler`` ，
需要实现 ``format_func`` 、 ``tokenize_func`` 两个方法，该方法是对加载到的每条数据进行预处理，可以参考 `alpaca_handler.py` 。

```python
@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class XXXInstructDataHandler(BaseInstructDataHandler):

    def format_func(self, example):
        # 自定义数据格式转换

    def tokenize_func(self, example):
        # 自定义tokenizer分词处理
```

``BaseInstructDataHandler`` 默认提供的实现了入口 ``handler`` 方法，用于遍历每条数据进行数据的预处理，
``format_func`` 用于实现如何从原始数据中转换成所需要的数据格式，而 ``tokenize_func`` 方法用于把处理后的数据进行按自定义分词，
实例里的入参 ``example`` 为获取到的每一条样本数据。

- 开发样例二

若用户想直接对于整个 dataset 进行数据处理，而不是每条数据分批处理的话，可以在自定义 handler 实现入口 ``handle`` 方法，得到的就是完整的 dataset，参考如下：

```python
    def handle(self, dataset):
        """data handler"""
        return dataset.rename_columns({"content":"prompt","summary":"answer"})
```

- alpaca 数据集示例

修改任务配置文件 [finetune_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/finetune_llama2_7b.yaml)。

修改如下参数：

```yaml
train_dataset: &train_dataset
  input_columns: &input_columns ["input_ids", "labels"]
  data_loader:
    type: CommonDataLoader
    shuffle: True
    split: "train"
    path: "llm-wizard/alpaca-gpt4-data"
    handler:
      - type: AlpacaInstructDataHandler
        tokenizer_name: llama2_7b
        seq_length: 4096
        prompt_key: "conversations"
        output_columns: *input_columns
  seed: 0
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  repeat: 1
  numa_enable: False
  prefetch_size: 1
```

其余参数介绍可以参考 [配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/appendix/conf_files.html) 的 “模型训练配置” 和 “模型评估配置”。

自定义数据 handler：

```python
@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class AlpacaInstructDataHandler(BaseInstructDataHandler):

    def format_func(self, example):
        """format func"""
        source = PROMPT_INPUT.format_map(example) \
            if example.get(self.input_key, "") != "" \
            else PROMPT_NO_INPUT.format_map(example)
        target = example.get(self.output_key)
        formatted_example = [
            {
                "from": self.user_role,
                "value": source,
            },
            {
                "from": self.assistant_role,
                "value": target,
            },
        ]

        return formatted_example

    def tokenize_func(self, messages):
        """tokenize func"""
        conversation = self.gen_prompt(messages)
        sep = self.template.sep + self.assistant_role + ": "
        # Tokenize conversations
        rounds = conversation.split(self.template.sep2)
        ids = [self.tokenizer.bos_token_id]
        mask = [1]
        for _, rou in enumerate(rounds):
            if rou == "":
                break
            conv_out = self.tokenizer(rou)
            ids.extend(conv_out['input_ids'][1:])
            mask.extend(conv_out['attention_mask'][1:])
        d = {'input_ids': ids, 'attention_mask': mask}
        # pylint: disable=W0212
        if not self.dynamic:
            d = self.tokenizer._pad(d, max_length=self.seq_length + 1, padding_strategy='max_length')
        input_id = d['input_ids'][:self.seq_length + 1]
        target = np.array(d['input_ids'])
        total_len = int(np.not_equal(target, self.tokenizer.pad_token_id).sum())
        cur_len = 1
        target[:cur_len] = self.ignore_token_id
        for _, rou in enumerate(rounds):
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(self.tokenizer(rou)['input_ids']) - 1
            instruction_len = len(self.tokenizer(parts[0])['input_ids']) - 3

            target[cur_len: cur_len + instruction_len] = self.ignore_token_id

            cur_len += round_len
        if self.dynamic:
            return {
                "input_ids": input_id,
                "labels": target[:len(input_id)].tolist()
            }
        target[cur_len:] = self.ignore_token_id
        if cur_len < self.seq_length + 1:
            if cur_len != total_len:
                target[:] = self.ignore_token_id
        else:
            target = target[:self.seq_length + 1]
        label = target.tolist()
        return {
            "input_ids": input_id,
            "labels": label,
        }
```

- ADGEN 数据集示例

修改任务配置文件 [run_glm3_6b_finetune_2k_800T_A2_64G.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/glm3/run_glm3_6b_finetune_2k_800T_A2_64G.yaml)。

修改如下参数：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: CommonDataLoader
    path: "HasturOfficial/adgen"
    split: "train"
    shuffle: True
    handler:
      - type: AdgenInstructDataHandler
    phase: "train"
    version: 3
    column_names: ["prompt", "answer"]
  tokenizer:
    type: ChatGLM3Tokenizer
    vocab_file: "/path/to/tokenizer.model"
  input_columns: ["input_ids", "labels"]
  max_source_length: 1024
  max_target_length: 1023
  ignore_pad_token_for_loss: True
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  batch_size: 8
  repeat: 1
  numa_enable: False
  prefetch_size: 1
  seed: 0
```

其余参数介绍可以参考 [配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/appendix/conf_files.html) 的 “模型训练配置” 和 “模型评估配置”。

自定义 adgen_handler：

```python
@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class AdgenInstructDataHandler(BaseInstructDataHandler):
    """agden data handler"""
    def handle(self, dataset):
        """data handler"""
        return dataset.rename_columns({"content": "prompt", "summary": "answer"})
```

#### 数据集packing

在`CommonDataLoader`中配置`PackingHandler`可以实现对数据进行packing处理，目前需要在前置处理中将原始数据处理为可输入模型的`input_ids`以及`labels`。

- 参数说明

| 参数名            | 概述                                                                                                                         |  类型  |
|----------------|----------------------------------------------------------------------------------------------------------------------------|:----:|
| type           | 固定为`PackingHandler`，该模块支持对数据进行packing，在[dataloader](#dataloader参数说明)中配置`packing=pack`和`packing=truncate`时，分别对数据进行非截断和截断的拼接 | str  |
| seq_length     | packing处理后数据的最大序列长度                                                                                                        | int  |
| pad_token      | 当packing后样本未达到最大长度时，对`input_ids`填充使用的token id，默认值为0                                                                        | int  |
| ignore_token   | 当packing后样本未达到最大长度时，对`labels`填充使用的token id，默认值为-100                                                                        | int  |

- packing示例

按照如下配置，对`alpaca`数据集进行预处理，即可实现在线packing。

```yaml
train_dataset: &train_dataset
  input_columns: &input_columns ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
  construct_args_key: *input_columns
  data_loader:
    type: CommonDataLoader
    shuffle: False
    split: "train"
    path: "llm-wizard/alpaca-gpt4-data"
    packing: pack
    handler:
      - type: AlpacaInstructDataHandler
        tokenizer_name: llama2_7b
        seq_length: 4096
        prompt_key: "conversations"
        output_columns: ["input_ids", "labels"]
      - type: PackingHandler
        seq_length: 4096
        output_columns: ["input_ids", "labels", "actual_seq_len"]
    adaptor_config:
       compress_mask: False
  seed: 0
  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  repeat: 1
  numa_enable: False
  prefetch_size: 1
```

使用上述配置文件处理`alpaca`数据集，会执行如下流程：

1. 使用`AlpacaInstructDataHandler`以及`llama2_7b`的`tokenizer`将原始文本数据处理为`input_ids`和`labels`；
2. 使用`PackingHandler`对处理后的`input_ids`和`labels`进行packing处理，得到拼接到`seq_length`长度的`input_ids`和`labels`, `actual_seq_len`拼接后样本中每个子样本的序列长度，在训练中会根据这个参数生成对应的数据掩码；
3. 如果在`adaptor_config`中设置`compress_mask=False`表示训练时返回完整的数据掩码，否则返回`actual_seq_len`；

#### 数据集离线处理

`CommonDataLoader`除了支持数据集在线加载与处理，还支持离线处理数据集并进行保存。

使用[datasets_preprocess.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/toolkit/data_preprocess/huggingface/datasets_preprocess.py)脚本可以离线处理 HuggingFace 数据集并进行保存。

- 参数说明

| 参数名           | 概述                                                        | 类型  |
|---------------|-----------------------------------------------------------|:---:|
| config        | 离线处理数据的配置文件，与在线处理使用方法相同，具体参考[dataloader](#dataloader参数说明) | str |
| save_path     | 数据集经过预处理后的保存路径                                            | str |
| register_path | 模型API的注册路径，其中包含模型相关Python文件，通常是research目录下模型文件夹的路径        | int |

- 使用示例

使用[数据集packing](#数据集packing)中提供的packing示例的配置文件即可，执行如下命令。

```shell
python toolkit/data_preprocess/huggingface/datasets_preprocess.py \
  --config data_process.yaml \
  --save_path /path/processed_data
```

如果需要加载保存后的数据集，需要对yaml进行如下修改：

```yaml
train_dataset: &train_dataset
  input_columns: &input_columns ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
  construct_args_key: *input_columns
  data_loader:
    type: CommonDataLoader
    shuffle: False
    load_func: "load_from_disk"
    path: "/path/processed_data"
    adaptor_config:
       compress_mask: False
```
