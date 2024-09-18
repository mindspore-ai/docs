# Dataset

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/function/dataset.md)

## MindRecord Dataset

MindRecord is an efficient data format developed by MindSpore for storing machine learning or deep learning datasets.

The MindRecord format is designed to improve data processing efficiency, especially in large-scale data training scenarios where data can be loaded and processed faster.
MindRecord files typically contain the input samples needed for model training, which are preprocessed (e.g., encoded, normalized) to optimize read speed and memory usage.

For more information about the implementation of MindRecord related interfaces and examples, please refer to the [documentation about MindRecord in MindSpore](https://www.mindspore.cn/docs/en/master/api_python/mindspore.mindrecord.html).

### How to Make a MindRecord Dataset

The MindRecord module provides methods to convert different datasets into MindRecord format.
You can use the FileWriter interface provided by MindSpore to generate MindRecord format datasets.

The following is an example of a MindRecord dataset based on a json format file:

1. Read json file

   ```python
   import json

   raw_data = None
   file = open("my_json_file.Json", "r")  # Open json file
   if file is not None:
      raw_data = json.load(file)  # Read json file into raw_data
      file.close()
   ```

2. Define a MindRecord ``schema`` and create a ``FileWriter`` object;

    ```python
    from mindspore.mindrecord import FileWriter

    # Define a schema for MindRecord
    schema = {'input_ids': {"type": "int32", "shape": [-1]}, 'labels': {"type": "int32", "shape": [-1]}}
    # Create a FileWriter object
    writer = FileWriter(file_name="output_file", shard_num=1)
    writer.add_schema(schema, "dataset_type")
    ```

3. Iterate through each Q&A pair in the processed json file, convert it to MindRecord format, and write it to a MindRecord file.

    ```python
    from internlm_tokenizer import InternLMTokenizer

    def tokenize_json(tokenizer, raw_data):
    """tokenize json file dataset"""
        content = [] # Read each json data and get its “input_ids”.
        for line in raw_data:
            stripped_line = line.strip()
            if stripped_line:
                line_ids = tokenizer(stripped_line)["input_ids"]
                content.append(line_ids)

        for ids in content:
            sample = {}
            sample['input_ids'] = np.array(ids, dtype=np.int32)
            yield sample

    # Tokenize the text data
    word_tokenizer = LlamaTokenizer(vocab_file=r"my_tokenizer.model")

    # Iterate through each Q&A pair in the processed json file, convert it to MindRecord format and write it to the MindRecord file
    # tokenize_json is a custom method to tokenize the dialog data in json.
    for x in tokenize_json(word_tokenizer, raw_data):
        writer.write_raw_data([x])
    writer.commit()
    ```

For the detailed cases, refer to [Examples of Data Preprocessing in Llama2](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)。

### Using MindRecord Format Datasets in Tasks

You can make a training or evaluation task use a prepared MindRecord format dataset by configuring dataset-related parameters in the yaml configuration file.

Here, as an example, for the Llama2-7B model pretraining task, the default configuration parameters and descriptions in the [pretrain_llama2_7b.yaml file](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/pretrain_llama2_7b.yaml#L39) are as follows:

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

Configure the following parameters to use MindRecord format datasets:

- data_loader.type: The type of the dataloader, which needs to be set to `MindDataset`.
- data_loader.dataset_dir: The path to the dataset file.
- input_columns: Sets the data columns for the input of the training dataset. Currently a pre-training scenario, set to `["input_ids"]`.

The rest of the parameters can be described in "model training configuration" and "model evaluation configuration [Configuration File Description](https://www.mindspore.cn/mindformers/docs/en/dev/appendix/conf_files.html).

## Online Dataset

Access to [modelers](https://modelers.cn/datasets) and [HuggingFace repository](https://huggingface.co/datasets), loading datasets online and expanding dataset sources.

### Docking HuggingFace Open Source Community

1. Environmental preparations

      The `HF_ENDPOINT` environment variable controls the actual remote repository used by the open source community HuggingFace. When not configured, it defaults to `https://huggingFace.co`, and for domestic environments, it needs to be configured to the mirror address ``export HF_ENDPOINT=https://hf- mirror.com``.

2. Installing dependencies

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

### Docking Modelers Open Source Community

1. Environmental preparations

   The `OPENMIND_HUB_ENDPOINT` environment variable controls the actual remote repository used by the Modelers Open Source community, and defaults to ``export OPENMIND_HUB_ENDPOINT=https://telecom.openmind.cn`` when not configured.

2. Installing dependencies

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

> When the openmind-datasets component is installed in the environment, the default interface is the Modelers open source community, if you want to interface with HuggingFace, the environment variable `USE_OM` can control which community to interface with, the default value is `ON` for the Modelers community, change it to `OFF` to interface with the HuggingFace community.

### Custom Data handler

Users can use custom data handler logic to perform various data preprocessing customization logic on loaded remote datasets.

#### Parameters

- type: The name of the custom data handler, which must inherit from ``BaseInstructDataHandler``.
- tokenizer_name: name of the tokenizer used.
- seq_length: length of the sequence.
- output_columns: output columns after data preprocessing.
- prompt_key: the name of the column after adding prompt processing.
- tokenizer: the configuration parameter of the tokenizer, it can be a dictionary or a string, or you can directly configure the ``tokenizer`` object.

#### Development Sample 1

The custom data handler is usually placed in the `mindformers/dataset/handler` directory, and the customized one needs to inherit the abstract base class ``BaseInstructDataHandler``.
You need to implement ``format_func`` and ``tokenize_func`` methods, which preprocess each data loaded. Refer to ``alpaca_handler.py``.

```python
@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class XXXInstructDataHandler(BaseInstructDataHandler):

    def format_func(self, example):
        # Custom data format conversion

    def tokenize_func(self, example):
        # Custom tokenizer split word processing
```

The ``BaseInstructDataHandler`` provides an implementation of the entry ``handler`` method by default, which is used to iterate over each piece of data for data preprocessing.
The ``format_func`` is used to implement how to convert the raw data into the desired data format, and the ``tokenize_func`` method is used to take the processed data and perform a customized tokenization.
The input parameter ``example`` in the example is each of the samples obtained.

#### Development Sample 2

If you want to process the data directly for the whole dataset instead of processing each piece of data in batches, you can implement the entry ``handle`` method in custom handler, and you will get the complete dataset, as shown below:

```python
    def handle(self, dataset):
        """data handler"""
        return dataset.rename_columns({"content":"prompt","summary":"answer"})
```

### alpaca Dataset Sample

#### Training Processes Loaded Directly from the Remote Repository

Modify the task configuration file [finetune_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/finetune_llama2_7b.yaml).

Modify the following parameters:

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

Configure the following parameters to use the alpaca dataset:

- input_columns: column names of the input data.
- data_loader.type: name of the class for data loading processing.
- data_loader.shuffle: whether the dataset is shuffled.
- data_loader.path: the remote path to load the dataset.
- data_loader.input_columns: which field conversions are used when datasets are converted to ms.datasets.
- data_loader.handler: data preprocessor class configuration, no data processing when empty.
- data_loader.handler.type: the class name of the data preprocessor class.
- data_loader.handler.tokenizer_name: name of the tokenizer.
- data_loader.handler.seq_length: length of the sequence.
- data_loader.handler.prompt_key: name of the data column after adding prompt processing.
- data_loader.handler.output_columns: columns to be output after data preprocessing.

The rest of the parameters can be described in "model training configuration" and "model evaluation configuration [Configuration File Description](https://www.mindspore.cn/mindformers/docs/en/dev/appendix/conf_files.html).

Custom data handler：

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

### ADGEN Dataset Sample

#### Training Processes Loaded Directly from the Remote Repository

Modify the task configuration file [run_glm3_6b_finetune_2k_800T_A2_64G.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/glm3/run_glm3_6b_finetune_2k_800T_A2_64G.yaml).

Modify the following parameters:

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
    vocab_file: "/data/z00827078/GLM3/tokenizer.model"
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

Configure the following parameters to use the ADGEN dataset:

- data_loader.type: class name of the data loading process.
- data_loader.path: path of the loaded dataset.
- data_loader.shuffle: whether to shuffle the dataset.
- data_loader.split: subset of the dataset, default is train set.
- data_loader.input_columns: which fields to use when converting datasets to ms.datasets.
- data_loader.handler: custom data handler.
- data_loader.handler.type: name of the type of the custom data handler.
- data_loader.handler.output_columns: the names of the dataset columns that will be output after processing.

The rest of the parameters can be described in "model training configuration" and "model evaluation configuration [Configuration File Description](https://www.mindspore.cn/mindformers/docs/en/dev/appendix/conf_files.html).

Custom adgen_handler:

```python
@MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
class AdgenInstructDataHandler(BaseInstructDataHandler):
    """agden data handler"""
    def handle(self, dataset):
        """data handler"""
        return dataset.rename_columns({"content":"prompt","summary":"answer"})
```