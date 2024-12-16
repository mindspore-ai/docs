# 数据集

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.10/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.4.10/docs/mindformers/docs/source_zh_cn/function/dataset.md)

## MindRecord 数据集

MindRecord 是由 MindSpore 开发的一种高效数据格式，用于存储机器学习或深度学习的数据集。

MindRecord 格式旨在提高数据处理效率，尤其是在大规模数据训练场景下，可以更快地加载和处理数据。
MindRecord 文件通常包含了模型训练所需的输入样本，这些样本经过预处理（如编码、归一化等），以优化读取速度和内存使用。

更多关于 MindRecord 相关接口的实现及案例，请参考 [MindSpore 中关于 《MindRecord》 的相关文档](https://www.mindspore.cn/docs/zh-CN/r2.4.10/api_python/mindspore.mindrecord.html)

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

详细案例可以参考 [Llama2 中的数据预处理案例](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)。

### 在任务中使用 MindRecord 格式数据集

通过在 yaml 配置文件中配置数据集相关参数，可以让训练或评测任务使用准备好的 MindRecord 格式数据集。

此处，以 Llama2-7B 模型预训练任务来举例说明，在 [pretrain_llama2_7b.yaml 文件](https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/llama2/pretrain_llama2_7b.yaml#L39) 中的默认配置参数及说明如下：

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

其余参数介绍可以参考 [配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/appendix/conf_files.html) 的 “模型训练配置” 和 “模型评估配置”。

## BIN 格式数据集

在大模型训练过程中，使用二进制格式（BIN格式）的数据集可以带来显著的性能和效率提升。当前 MindFormers 框架也适配了对 BIN 格式数据集的处理能力，包括如何制作 BIN 格式数据集和在任务中使用 BIN 格式数据集。

### 如何制作 BIN 格式数据集

当前 MindFormers 提供的预处理脚本仅支持处理 json 格式的文件，需要用户在使用预处理脚本前将原始数据集的文件格式转换成符合预处理脚本支持的 json 格式的文件，支持的 json 格式的文件格式如下：

```json
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
```

以 Llama2 处理 Wiki数据集为例，原始Wiki数据集的下载参考 [Llama2 中的数据预处理案例](https://gitee.com/mindspore/mindformers/blob/r1.3.0/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)，在处理成符合预处理脚本支持格式的数据集后，直接调用 [mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py](https://gitee.com/mindspore/mindformers/blob/r1.3.0/mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py)，具体命令如下：

```shell
python mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py \
--input /path/to/wiki.json \
--output-prefix /path/to/my_wiki_1024 \
--tokenizer-type LlamaTokenizer \
--vocab-file /path/to/tokenizer.model \
--add_bos_token True \
--add_eos_token True \
--pad_or_stitch stitch \
--seq-length 1024 \
--workers 1
```

预处理脚本的入参说明如下：

- input: 待处理的数据集处理成 json 格式后的文件路径
- output-prefix: 预处理后的输出文件的文件名前缀
- tokenizer-type: 模型对应的 tokenizer 的类型
- vocab-file: 模型的 tokenizer.model 或者其他格式的 vocab file
- add_bos_token: 是否在数据的首位置添加 bos_token，默认 False，具体设置参考各个模型要求
- add_eos_token: 是否在数据的末位置添加 eos_token，默认 False，具体设置参考各个模型要求
- pad_or_stitch: 根据训练任务的要求，设置是否拼接还是补齐，pad 为补齐，stitch 为拼接
- seq-length: 数据集处理的数据长度，需用户自行设置
- workers: 预处理时并行 worker 的数量

执行以上命令之后，会得到两个文件，分别为 .bin 和 .idx 格式的文件。

### 在任务中使用 BIN 格式数据集

通过在 yaml 配置文件中配置数据集相关参数，可以让训练任务使用准备好的 BIN 格式数据集。

此处，以 Llama2-7B 模型预训练任务来举例说明，在 [pretrain_llama2_7b.yaml 文件](https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/llama2/pretrain_llama2_7b.yaml#L39) 中的配置参数的修改及说明如下：

```yaml
# dataset
train_dataset: &train_dataset
  data_loader:
    type: IndexedDataLoader
    path_prefix: ""
    shuffle: False
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

配置如下参数以使用 BIN 格式数据集：

- data_loader.type：dataloader 的类型，此处需要设置为 `IndexedDataLoader` 。
- data_loader.path_prefix：数据集文件名的前缀。
- input_columns：设置训练数据集输入的数据列。当前为预训练场景，设置为 `["input_ids"]` 。

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

修改任务配置文件 [finetune_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/llama2/finetune_llama2_7b.yaml) 。

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

其余参数介绍可以参考 [配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/appendix/conf_files.html) 的 “模型训练配置” 和 “模型评估配置”。

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

修改任务配置文件 [run_glm3_6b_finetune_2k_800T_A2_64G.yaml](https://gitee.com/mindspore/mindformers/blob/r1.3.0/configs/glm3/run_glm3_6b_finetune_2k_800T_A2_64G.yaml) 。

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

其余参数介绍可以参考 [配置文件说明](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.3.2/appendix/conf_files.html) 的 “模型训练配置” 和 “模型评估配置”。

自定义 adgen_handler：

```python
@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class AdgenInstructDataHandler(BaseInstructDataHandler):
    """agden data handler"""
    def handle(self, dataset):
        """data handler"""
        return dataset.rename_columns({"content":"prompt","summary":"answer"})
```