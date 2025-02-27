# 数据集

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/function/dataset.md)

目前MindSpore Transformers的预训练和微调支持多种格式的数据集加载能力，包括Megatron多源数据集、MindRecord数据集以及在线数据集的加载方式。每种格式的数据集的具体使用方法的参考如下。

## Megatron多源数据集

Megatron多源数据集是指从多个不同来源收集的数据集，这些数据集可以包含不同的文本类型、格式和领域。使用多源数据集可以帮助模型学习到更广泛的语言特征和知识，从而提高模型的泛化能力和性能。Megatron框架目前实现的多源数据集，需要先将原数据集预处理成BIN格式的数据集。当前MindSpore Transformers已经原生适配了Megatron多源数据集，提供了制作BIN格式数据集的脚本，支持在训练任务中直接使用Megatron多源数据集。

### 制作 BIN 格式数据集

MindSpore Transformers 提供了一个预处理脚本 [mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py) 将文本数据转换成BIN格式数据集，该脚本当前仅支持处理特定形式的 JSON 格式的文件。用户需要先将原始数据集文件转换成特定形式的JSON格式的文件，再使用预处理脚本生成BIN格式的数据集文件。当前 MindSpore Transformers 中的一些模型已经提供了将特定开源数据集转换成特定形式 JSON 格式文件的脚本，用户如想使用自有数据集，则需要通过自行编写脚本的方式将其转换为所需形式。

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

   原始 Wiki 数据集的下载参考 [Llama2 数据集下载](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)。

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

    处理成上述这样特定的 JSON 格式的文件后，调用 [mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py](https://gitee.com/mindspore/mindformers/blob/dev/mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py) 将其转换成BIN格式的数据集，具体命令如下：

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
       "compute_communicate_fusion_level": 3,
       "dataset_broadcast_opt_level": 3
   }
   ```

2. 设置环境变量

    在命令行输入如下命令设置环境变量：

    ```shell
    export MS_DEV_DYNAMIC_SINK1=False
    ```

3. 修改训练任务的 YAML 配置文件

    在 YAML 配置文件中配置Megatron多源数据集的相关参数。此处，以 Llama2-7B 模型预训练任务来举例说明，修改 [pretrain_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/pretrain_llama2_7b.yaml#L39) 中的 `train_dataset` 、 `runner_config` 、 `parallel_config` 、 `parallel` 以及 `context` 配置项。具体修改及说明如下：

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
        input_columns: ["input_ids", "labels", "loss_mask", "position_ids"]
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

    当前多源数据集目前还存在限制，仅支持非 full batch 的场景，需要根据以下对相应配置项进行修改：

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

    - parallel.dataset_strategy：仅支持 List of List 类型，List中子List的个数需要等于 train_dataset.input_columns 的长度，并且 List 中的每个子 List 需要和数据集返回的数据的shape保持一致。一般在数据的第1维进行数据并行切分，所以子List的第1位数配置成 `*dp` ，其他位配置为 `1` 。具体原理可以参考[数据集切分](https://www.mindspore.cn/docs/zh-CN/master/model_train/parallel/dataset_slice.html)。

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

更多关于 MindRecord 相关接口的实现及案例，请参考 [MindSpore 中关于 《MindRecord》 的相关文档](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mindrecord.html)

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

详细案例可以参考 [Llama2 中的数据预处理案例](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)。

### 在任务中使用 MindRecord 格式数据集

通过在 yaml 配置文件中配置数据集相关参数，可以让训练或评测任务使用准备好的 MindRecord 格式数据集。

此处，以 Llama2-7B 模型预训练任务来举例说明，在 [pretrain_llama2_7b.yaml 文件](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/pretrain_llama2_7b.yaml#L39) 中的默认配置参数及说明如下：

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

其余参数介绍可以参考 [配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/appendix/conf_files.html) 的 “模型训练配置” 和 “模型评估配置”。

## 在线数据集

接入 [魔乐仓库](https://modelers.cn/datasets)、[HuggingFace 仓库](https://huggingface.co/datasets)，在线加载数据集，扩大数据集来源。

### 对接 HuggingFace 开源社区

1. 环境准备

      环境变量 `HF_ENDPOINT` 可以控制开源社区huggingFace实际使用的远程仓库，未配置时默认为 `https://huggingFace.co` ，
      针对国内环境，需要配置成镜像地址 ```export HF_ENDPOINT=https://hf-mirror.com``` 。

2. 安装依赖

   ```shell
   git clone https://gitee.com/openmind-ai/openmind-hub.git
   cd openmind-hub
   pip install -e .
   cd ..
   pip install datasets==2.18.0
   git clone https://gitee.com/openmind-ai/openmind-extension-for-datasets.git
   cd openmind-extension-for-datasets
   pip install -e .
   cd ..
   ```

### 对接魔乐开源社区

1. 环境准备

   环境变量 `OPENMIND_HUB_ENDPOINT` 可以控制魔乐开源社区实际使用的远程仓库，
   未配置时默认为 ```export OPENMIND_HUB_ENDPOINT=https://telecom.openmind.cn``` 。

2. 安装依赖

   ```shell
   git clone https://gitee.com/openmind-ai/openmind-hub.git
   cd openmind-hub
   pip install -e .
   cd ..
   pip install datasets==2.18.0
   git clone https://gitee.com/foundation-models/openmind-datasets.git
   cd openmind-datasets
   pip install -e .
   cd ..
   ```

> 当环境安装了 openmind-datasets 三方件时，默认对接的是魔乐开源社区，如果这是想对接 HuggingFace，环境变量 `USE_OM` 可以控制具体对接哪个社区，默认值为 `ON` 为魔乐社区，修改为 `OFF` 对接 HuggingFace 社区

### 自定义数据 handler

用户可以使用自定义数据 handler 逻辑，对加载到的远端数据集进行各种数据预处理定制逻辑。

#### 参数

- type：自定义数据 handler 名称，自定义 handler 必须继承 ``BaseInstructDataHandler`` 。
- tokenizer_name：使用的 tokenizer 分词器名称。
- seq_length：序列长度。
- output_columns：数据预处理后输出的数据列。
- prompt_key：增加 prompt 处理后数据列名称。
- tokenizer：tokenizer 配置参数, 可以是字典或者字符串，也可以直接配置 ``tokenizer`` 对象。

#### 开发样例一

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

#### 开发样例二

若用户想直接对于整个 dataset 进行数据处理，而不是每条数据分批处理的话，可以在自定义 handler 实现入口 ``handle`` 方法，得到的就是完整的 dataset，参考如下：

```python
    def handle(self, dataset):
        """data handler"""
        return dataset.rename_columns({"content":"prompt","summary":"answer"})
```

### alpaca 数据集示例

#### 训练流程直接从远端仓库加载

修改任务配置文件 [finetune_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/finetune_llama2_7b.yaml) 。

修改如下参数：

```yaml
train_dataset:
  input_columns: &input_columns ["input_ids", "labels"]
  data_loader:
    type: CommonDataLoader
    shuffle: True
    split: "train"
    path: "AI_Connect/alpaca"
    input_columns: *input_columns
    handler:
      type: AlpacaInstructDataHandler
      tokenizer_name: llama2_13b
      seq_length: 4096
      prompt_key: "conversations"
      output_columns: *input_columns
```

配置如下参数以使用 alpaca 数据集：

- input_columns：输入的数据的列名。
- data_loader.type：数据加载处理的类名。
- data_loader.shuffle：数据集是否打乱。
- data_loader.path：加载数据集的远端路径。
- data_loader.input_columns：datasets 转换为 ms.datasets 时，使用哪些字段转换。
- data_loader.handler：数据预处理类配置，为空时不做数据处理。
- data_loader.handler.type：数据预处理类的类名。
- data_loader.handler.tokenizer_name：分词器名称。
- data_loader.handler.seq_length：序列长度。
- data_loader.handler.prompt_key：增加 prompt 处理后数据列名称。
- data_loader.handler.output_columns：数据预处理后输出的数据列。

其余参数介绍可以参考 [配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/appendix/conf_files.html) 的 “模型训练配置” 和 “模型评估配置”。

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
        d = self.tokenizer._pad(d, max_length=self.seq_length + 1, padding_strategy='max_length')
        input_id = d['input_ids'][:self.seq_length + 1]
        # attention_mask.append(d['attention_mask'])
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

### ADGEN 数据集示例

#### 训练流程直接从远端仓库加载

修改任务配置文件 [run_glm3_6b_finetune_2k_800T_A2_64G.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/glm3/run_glm3_6b_finetune_2k_800T_A2_64G.yaml) 。

修改如下参数：

```yaml
train_dataset: &train_dataset
  data_loader:
    type: CommonDataLoader
    path: "xxx/ADGEN"
    split: "train"
    shuffle: True
    input_columns: ["prompt", "answer"]
    handler:
      type: AdgenInstructDataHandler
      output_columns: ["content", "summary"]
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
  phase: "train"
  version: 3
  seed: 0
```

配置如下参数以使用 ADGEN 数据集：

- data_loader.type：数据加载处理的类名。
- data_loader.path：加载数据集路径。
- data_loader.shuffle：数据集是否打乱。
- data_loader.split：数据集子集，默认加载 train 集。
- data_loader.input_columns：datasets 转换为 ms.datasets 时，使用哪些字段转换。
- data_loader.handler：自定义数据处理器。
- data_loader.handler.type：自定义数据处理器类型名称。
- data_loader.handler.output_columns：处理完后输出的数据集列名。

其余参数介绍可以参考 [配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/appendix/conf_files.html) 的 “模型训练配置” 和 “模型评估配置”。

自定义 adgen_handler：

```python
@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class AdgenInstructDataHandler(BaseInstructDataHandler):
    """agden data handler"""
    def handle(self, dataset):
        """data handler"""
        return dataset.rename_columns({"content":"prompt","summary":"answer"})
```