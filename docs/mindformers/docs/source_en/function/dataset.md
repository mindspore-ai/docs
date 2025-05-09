# Dataset

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/docs/mindformers/docs/source_en/function/dataset.md)

At present, MindSpore Transformers' pre-training and fine-tuning support the ability to load datasets in multiple formats, including loading methods for Megatron Dataset, MindRecord Dataset, and HuggingFace datasets. The specific usage instructions for each format of dataset are as follows.

## Megatron Dataset

Megatron Dataset refers to a dataset collected from multiple different sources, it contains different text types, formats, and domains. Using dataset can help models learn a wider range of language features and knowledge, thereby improving their generalization ability and performance. The current implementation of the Megatron framework requires preprocessing the original dataset into a BIN format dataset. MindSpore Transformers have been natively adapted to the Megatron Dataset, providing scripts for creating BIN format datasets and supporting direct use of the Megatron Dataset in training tasks.

### How to Make a BIN Format Dataset

MindSpore Transformers provides a preprocessing script [mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py), which can convert text data to a BIN format dataset. This script currently only supports processing files in a specific JSON format. Users need to first convert the original dataset file into a specific JSON format file, and then use a preprocessing script to generate a BIN format dataset file. Some models in MindSpore Transformers currently provide scripts for converting specific open-source datasets into JSON format files. If users want to use their own datasets, they need to write their own scripts to convert them into the desired format.

The format of the required JSON format file content is as follows:

```json
{"id": "0", "text": "The quick brown fox", "type": "Eng", "src": "www.nvidia.com", "title": "First Part"}
{"id": "1", "text": "jumps over the lazy dog", "type": "Eng", "src": "The Internet", "title": "Second Part"}
...
```

Each piece of data consists of several key value pairs, and the supported keys and descriptions are as follows:

- `"id"`: The numbering of the data should be in order, required
- `"text"`: Text data actually used for training, required
- `"type"`: Indicate language type, optional
- `"src"`: Indicate the source of the data, optional
- `"title"`: Indicate the title of the data, optional

Taking the processing of Wiki datasets and their use as pre-training for Llama2 models as an example, the detailed steps for creating BIN format datasets are explained below:

1. Download Wiki Dataset

   For the original Wiki Dataset downloading, refer to [Llama2 Dataset Download](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87).

2. Generate JSON Format File

   The original format of the Wiki Dataset is as follows:

   ![](image/wikitext_sample.png)

   The format of the JSON file `wiki.json` after processing the Wiki Dataset is as follows (omitting long text):

   ```json
   {"id": 0, "text": "The gold dollar or gold one ..."}
   {"id": 1, "text": "Super Mario Land is a 1989 ..."}
   {"id": 2, "text": "The Sinclair Scientific Programmable ..."}
   ...
   ```

3. Download The Vocabulary File For Llama2

   In the preprocessing script, the raw text data will be processed into Tokens using the Tokenizer of the model, therefore, it is necessary to download the vocabulary file in advance.

   Download link for Llama2 vocabulary file: [tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

4. Generate BIN Format Files Using Preprocessing Scripts

    After processing into the specific JSON format file mentioned above, using [mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/mindformers/tools/dataset_preprocess/preprocess_indexed_dataset.py) to convert it into a BIN format dataset, the specific command is as follows:

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

    Configuration parameter description:

    - `--input`: Path to JSON format file
    - `--output-prefix`: The file name prefix of the preprocessed output file
    - `--tokenizer-type`: The type of tokenizer corresponding to the model
    - `--vocab-file`: The path of the vocabulary file for the tokenizer model tokenizer
    - `--add_bos_token`: Add bos_token at the beginning of the data, Default: False
    - `--add_eos_token`: Add eos_token at the ending of the data, Default: False
    - `--pad_or_stitch`: According to the requirements of the training task, set whether to splice or fill in, pad is in fill in mode, this mode will fill in the data with insufficient length to the seq length; Stitch is a concatenation mode that concatenates multiple pieces of data into data with a length of seq length
    - `--seq-length`: Preprocess the length of each piece of data
    - `--workers`: The number of parallel workers during preprocessing

After executing the above command, two files will be obtained, in `.bin` and `.idx` formats respectively. The `.bin` format file stores the actual data, and `.idx` format file stores the index of each piece of data.

### Using Megatron Datasets in Training Tasks

Use the Megatron multi-source dataset in the training task as follows:

1. Prepare the `parallel_speed_up.json` file

   `parallel_speed_up.json` is a dataset parallel communication configuration file, and the file content is as follows:

   ```json
   {
       "dataset_broadcast_opt_level": 3
   }
   ```

2. Set environment variables

    Enter the following command at the command line to set environment variables:

    ```shell
    export MS_DEV_DYNAMIC_SINK1=False
    ```

3. Modify YAML configuration files for training tasks

    Configure the relevant parameters of Megatron Dataset in YAML configuration file. Here, taking the Llama2-7B model pre-training task as an example, modify `train_dataset` , `runner_config` , `parallel_config` , `parallel` and `context` in  [pretrain_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/pretrain_llama2_7b.yaml#L39). The specific modifications and explanations are as follows:

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

    Among them:

    - data_loader.type: The type of dataloader, should be set to `BlendedMegatronDatasetDataLoader`.
    - data_loader.datasets_type: Dataset type, currently only supports `GPTDataset`.
    - data_loader.sizes: `- 1000` , `- 0` , `- 0` are the sampling sizes for the training set, test set, and validation set, respectively. Currently, only the training set can be configured.
    - input_columns: Set the input data columns for the training dataset, typically configured as `["input_ids", "labels", "loss_mask", "position_ids"]` .
    - data_loader.config.seed: Random number seed when creating a dataset. Default: `1234` .
    - data_loader.config.seq_length: The length of each piece of data must be consistent with the model.model_config.seq_length in the YAML configuration.
    - data_loader.config.split: Split string, separate the weights of the training set, test set, and validation set with commas, used to split the dataset when drawing samples from a single distribution. Currently, only supports configuration as `"1, 0, 0"` .
    - data_loader.config.data_path: The number is the weight of each dataset, and the string is the path of the dataset BIN file, which needs to remove the file format suffix `.bin` .
    - data_loader.config.num_dataset_builder_threads: The number of processes used when creating the dataset. Default: `1` .
    - data_loader.config.eod_mask_loss: Do you want to use the switch of eod mask. Default: `False` .
    - data_loader.config.create_attention_mask: Whether to construct attention_mask. Default: `True` .

    There are still limitations to the current Megatron Dataset, which only supports non full batch scenarios, and it does not support the parallel feature of seq_pipe. The corresponding configuration items need to be modified according to the following:

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

    The configuration instructions that need to be noted are as follows:

    - parallel.dataset_strategy: Only support List of List type, parallel.dataset_strategy: Only support List of List type. The number of sub lists in a List needs to be equal to the length of train_dataset.input_columns, and each sub List in the List needs to be consistent with the shape of the data returned by the dataset. Generally, parallel data partitioning is performed in the first dimension of the data, so the first bit of the sub List is configured as `*dp` , and the other bits are configured as `1` . The specific principle can be referred to [Dataset Segmentation](https://www.mindspore.cn/tutorials/en/r2.6.0/parallel/dataset_slice.html).

4. Compile Megatron Dataset module

    MindSpore Transformers have built-in Megatron Dataset module code, before starting the training task, the following command needs to be executed for compilation:

    ```shell
    pip install pybind11
    cd mindformers/dataset/blended_datasets
    make
    ```

## MindRecord Dataset

MindRecord is an efficient data format developed by MindSpore for storing machine learning or deep learning datasets.

The MindRecord format is designed to improve data processing efficiency, especially in large-scale data training scenarios where data can be loaded and processed faster.
MindRecord files typically contain the input samples needed for model training, which are preprocessed (e.g., encoded, normalized) to optimize read speed and memory usage.

For more information about the implementation of MindRecord related interfaces and examples, please refer to the [documentation about MindRecord in MindSpore](https://www.mindspore.cn/docs/en/r2.6.0/api_python/mindspore.mindrecord.html).

### How to Make a MindRecord Dataset

The MindRecord module provides methods to convert different datasets into MindRecord format.
You can use the FileWriter interface provided by MindSpore to generate MindRecord format datasets.

The following is an example of a MindRecord dataset based on a json format file, taking Llama2 as an example:

1. Prepara json file

   Prepare a json file like this, named `mydata.json`:

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

2. Read json file

   ```python
   import json

   raw_data = None
   file = open("mydata.json", "r")  # Open json file
   if file is not None:
      raw_data = json.load(file)  # Read json file into raw_data
      file.close()
   ```

3. Define a MindRecord ``schema`` and create a ``FileWriter`` object;

    ```python
    from mindspore.mindrecord import FileWriter

    # Define a schema for MindRecord
    schema = {'input_ids': {"type": "int32", "shape": [-1]}
    # Create a FileWriter object
    writer = FileWriter(file_name="output_file", shard_num=1)
    writer.add_schema(schema, "dataset_type")
    ```

4. Iterate through each piece of data in the processed json file, convert it to MindRecord format, and write it to a MindRecord file.

   Word list download link: [tokenizer.model](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/llama2/tokenizer.model)

    ```python
    import numpy as np
    from mindformers import LlamaTokenizer

    def tokenize_json(tokenizer, raw_data):
        """tokenize json file dataset"""
        content = [] # Read each json data and get its “input_ids”.
        for line in raw_data:
            stripped_line = line['text'].strip()
            if stripped_line:
                line_ids = tokenizer(stripped_line)["input_ids"]
                content.append(line_ids)

        for ids in content:
            sample = {}
            sample['input_ids'] = np.array(ids, dtype=np.int32)
            yield sample

    # Tokenize the text data
    word_tokenizer = LlamaTokenizer(vocab_file=r"tokenizer.model")

    # Iterate through each piece of data in the processed json file, convert it to MindRecord format and write it to the MindRecord file
    # tokenize_json is a custom method to tokenize the dialog data in json.
    for x in tokenize_json(word_tokenizer, raw_data):
        writer.write_raw_data([x])
    writer.commit()
    ```

For the detailed cases, refer to [Examples of Data Preprocessing in Llama2](https://gitee.com/mindspore/mindformers/blob/r1.5.0/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87).

### Using MindRecord Format Datasets in Tasks

You can make a training or evaluation task use a prepared MindRecord format dataset by configuring dataset-related parameters in the yaml configuration file.

Here, as an example, for the Llama2-7B model pretraining task, the default configuration parameters and descriptions in the [pretrain_llama2_7b.yaml file](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/pretrain_llama2_7b.yaml#L39) are as follows:

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

The rest of the parameters can be described in "model training configuration" and "model evaluation configuration [Configuration File Description](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/appendix/conf_files.html).

## HuggingFace Datasets

Currently, the dataset loading functionality has been integrated with the [ModelScope Open-Source Community](https://modelers.cn/datasets) and the [HuggingFace Community](https://huggingface.co/datasets), supporting online dataset loading and preprocessing. Additionally, datasets can be [packed](#dataset-packing) to enhance model training efficiency.

### Usage Instructions

HuggingFace datasets support online and offline loading of datasets from both the HuggingFace community and the MoLo open-source community. Below is an introduction to environment preparation, the dataset loading process, and how to configure the use of HuggingFace datasets in configuration files.

#### Integrating with Open-Source Communities

- Integrating with HuggingFace Community

   To use datasets from the HuggingFace community, follow these steps:

  1. Environment Setup

     The environment variable `HF_ENDPOINT` controls the remote repository used by HuggingFace. By default, it is set to `https://huggingFace.co`.
     For users in China, it is recommended to configure it to the mirror address ```export HF_ENDPOINT=https://hf-mirror.com``` .

  2. Install Dependencies

     ```shell
     pip install datasets
     ```

- Integrating with ModelScope Open-Source Community

   To use datasets from the ModelScope Open-Source Community, follow these steps:

   1. Environment Setup

      The environment variable `OPENMIND_HUB_ENDPOINT` controls the remote repository used by the ModelScope Open-Source Community.
      Defaults to ```export OPENMIND_HUB_ENDPOINT=https://telecom.openmind.cn``` when not configured.

   2. Install Dependencies

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

> When the openmind-datasets component is installed in the environment, the default interface is the Modelers open source community, if you want to interface with HuggingFace, the environment variable `USE_OM` can control which community to interface with, the default value is `ON` for the Modelers community, change it to `OFF` to interface with the HuggingFace community.

#### Dataset Loading Process

![commondataloader.png](../../source_zh_cn/function/image/commondataloader.png)

The online dataset loading and processing functionality is primarily implemented through `CommonDataLoader`. The data loading part can be customized via configuration files, with detailed configuration instructions available in the [dataloader parameter description](#dataloader-parameter-description). The online loading module requires users to implement customizations for different datasets. For example, the `AlpacaInstructDataHandler` class can be used to preprocess the `alpaca` dataset. For more information, please refer to [Custom Data Handler](#custom-data-handler).

#### dataloader Parameter Description

The online dataset loading feature is enabled by configuring the `data_loader` in the configuration file. Below is an example configuration for online dataset loading:

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

Parameter descriptions for `data_loader` are as follows:

| Parameter Name | Description                                                                                                                                                                                                                            | Type |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----:|
| type           | Fixed as `CommonDataLoader`. This module supports loading datasets from HuggingFace and the ModelScope open-source community.                                                                                                          | str  |
| packing        | Packing configuration when processing datasets with `handler`. Options include `pack` and `truncate`.                                                                                                                                  | str  |
| load_func      | The function used to load datasets. Options are `load_dataset` and `load_from_disk`. Use `load_from_disk` for data saved via the `save_to_disk` function, and `load_dataset` for other scenarios. The default value is `load_dataset`. | str  |
| path           | When `load_func=load_dataset`, this parameter aligns with the interface in [datasets.load_dataset](https://huggingface.co/docs/datasets/loading). When `load_func=load_from_disk`, it specifies the dataset loading path.              | str  |
| data_files     | When `load_func=load_dataset`, this parameter aligns with the interface in [datasets.load_dataset](https://huggingface.co/docs/datasets/loading). It is ineffective when `load_func=load_from_disk`.                                   | str  |
| handler        | Multiple `handlers` can be configured to preprocess the loaded dataset in the order specified. For details on `handler` configuration, refer to the handler parameter description in [Custom Data Handler](#custom-data-handler).      | list |
| adaptor_config | Dataset-related configuration during model training. Currently supports `compress_mask`, effective when `packing` is set. If enabled, it returns a compressed data mask. Default is `False`.                                           | dict |
| shuffle        | Indicates whether random sampling is enabled when loading the dataset.                                                                                                                                                                 | bool |
| column_names   | Specifies the column names returned by the dataset. If not set, all columns are returned.                                                                                                                                              | list |
| is_dynamic     | Indicates whether the dataset returns dynamic-length data. Default is `False`.                                                                                                                                                         | bool |

> In addition to the above configurations, all parameters from the [datasets.load_dataset](https://huggingface.co/docs/datasets/loading) interface are supported with the same meanings and functions.

When packing is configured, the dataset returns an `actual_seq_len` column. For more information, refer to the `actual_seq_qlen` and `actual_seq_kvlen` parameter descriptions in the [documentation](https://www.hiascend.com/document/detail/zh/Pytorch/600/ptmoddevg/trainingmigrguide/performance_tuning_0027.html).

### Feature Introduction

#### Dynamic Sequence Length Fine-Tuning

`CommonDataLoader` supports dynamic shape fine-tuning using HuggingFace datasets, which can be loaded online or offline. Below, we use the `alpaca` dataset as an example to demonstrate the configuration for dynamic shape fine-tuning.

- Online Loading

  The online dataset name is `llm-wizard/alpaca-gpt4-data`. You can search and download it from the [HuggingFace official website](https://huggingface.co/datasets) or load it directly using the online name.

  Example configuration for online loading:

  ```yaml
  train_dataset: &train_dataset
    input_columns: &input_columns ["input_ids", "labels"]
    dynamic_batch: True                    # Enable dynamic shape
    divisor: 32                            # With divisor and remainder configured, seq_length in dynamic shape will become a multiple of divisor and the sum of remainder
    remainder: 1
    data_loader:
      type: CommonDataLoader
      shuffle: True
      split: "train"                       # Subset name of the online dataset
      path: "llm-wizard/alpaca-gpt4-data"  # Online dataset name
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

  1. For parameter descriptions in `train_dataset`, please refer to the [documentation](https://www.mindspore.cn/mindformers/docs/zh-CN/r1.5.0/appendix/conf_files.html).

  2. `AlpacaInstructDataHandler` is an online processing script developed for the `alpaca` dataset. If using a different dataset, you need to implement a custom data handler by referring to the [Custom Data Handler](#custom-data-handler) guide.

- Offline Loading

  For offline loading, you need to prepare the JSON files of the `alpaca` dataset. The offline configuration differs from the online configuration only in the following parameters:

  ```yaml
   train_dataset:
     data_loader:
       path: "json"                               # loading datasets using the load_dataset interface
       data_files: '/path/alpaca_gpt4_data.json'  # the file path of the alpaca dataset
   ```

After configuring the dataset loading method, you also need to set `is_dynamic=True` in the model configuration to enable dynamic shape training for the model.

```yaml
model_config:
  is_dynamic: True
```

Since dynamic shapes may lead to operator compilation caching, it is recommended to set the following environment variables to limit the number of cached compilations when running in a memory-constrained environment. This helps prevent out-of-memory issues:

```shell
export ACLNN_CACHE_LIMIT=10
export MS_DEV_RUNTIME_CONF="aclnn_cache_queue_length:64"
```

- The `ACLNN_CACHE_LIMIT` parameter description can be found in the [documentation](https://www.hiascend.com/document/detail/zh/canncommercial/800/apiref/envvar/envref_07_0031.html).
- `MS_DEV_RUNTIME_CONF` is a parameter in MindSpore for setting the operator cache queue length. The value `64` represents the length of the sequence, which defaults to `1024`. This can be adjusted based on the actual environment. Setting the value too small may affect model training performance.

After completing all the configurations above, you can proceed with dynamic shape fine-tuning by referring to the documentation for the specific model you are using.

#### Custom Data Handler

Users can define custom data handlers to apply various preprocessing logic to the loaded dataset.

- Handler Parameter Description

  | Parameter Name | Description                                                                                                                           |   Type   |
  |----------------|---------------------------------------------------------------------------------------------------------------------------------------|:--------:|
  | type           | Custom data handler name. A custom handler must inherit from `BaseInstructDataHandler`.                                               |   str    |
  | tokenizer_name | Name of the tokenizer used.                                                                                                           |   str    |
  | tokenizer      | Tokenizer configuration parameters. Can be a dictionary, string, or a `tokenizer` object. Takes lower priority than `tokenizer_name`. | dict/str |
  | seq_length     | Maximum sequence length, usually the same as the model's sequence length.                                                             |   int    |
  | output_columns | Column names of the processed data returned after preprocessing.                                                                      |   list   |
  | prompt_key     | Column name for data after applying prompt processing.                                                                                |   str    |

- Development Sample 1

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

- Development Sample 2

  If you want to process the data directly for the whole dataset instead of processing each piece of data in batches, you can implement the entry ``handle`` method in custom handler, and you will get the complete dataset, as shown below:

  ```python
      def handle(self, dataset):
          """data handler"""
          return dataset.rename_columns({"content":"prompt","summary":"answer"})
  ```

- alpaca Dataset Sample

  Modify the task configuration file [finetune_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/llama2/finetune_llama2_7b.yaml).

  Modify the following parameters:

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

  The rest of the parameters can be described in "model training configuration" and "model evaluation configuration [Configuration File Description](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/appendix/conf_files.html).

  Custom data handler:

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

- ADGEN Dataset Sample

  Modify the task configuration file [run_glm3_6b_finetune_2k_800T_A2_64G.yaml](https://gitee.com/mindspore/mindformers/blob/r1.5.0/configs/glm3/run_glm3_6b_finetune_2k_800T_A2_64G.yaml).

  Modify the following parameters:

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

  The rest of the parameters can be described in "model training configuration" and "model evaluation configuration [Configuration File Description](https://www.mindspore.cn/mindformers/docs/en/r1.5.0/appendix/conf_files.html).

  Custom adgen_handler:

  ```python
  @MindFormerRegister.register(MindFormerModuleType.DATA_HANDLER)
  class AdgenInstructDataHandler(BaseInstructDataHandler):
      """agden data handler"""
      def handle(self, dataset):
          """data handler"""
          return dataset.rename_columns({"content": "prompt", "summary": "answer"})
  ```

#### Dataset Packing

Configuring `PackingHandler` in `CommonDataLoader` allows for packing processing of the data. Currently, the original data needs to be processed into `input_ids` and `labels` that can be fed into the model during the preprocessing step.

- Parameter Description

  | Parameter Name | Description                                                                                                                                                                                                                                                  | Type |
  |----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----:|
  | type           | Fixed as `PackingHandler`. This module supports packing data. When `packing=pack` or `packing=truncate` is configured in [dataloader](#dataloader-parameter-description), it performs non-truncating and truncating concatenation of the data, respectively. | str  |
  | seq_length     | Maximum sequence length of the data after packing.                                                                                                                                                                                                           | int  |
  | pad_token      | Token ID used for padding `input_ids` when the packed sample does not reach the maximum length. Default value is 0.                                                                                                                                          | int  |
  | ignore_token   | Token ID used for padding `labels` when the packed sample does not reach the maximum length. Default value is -100.                                                                                                                                          | int  |

- Packing Example

  By following the configuration below, the `alpaca` dataset can be preprocessed to achieve online packing.

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

Using the above configuration file to process the `alpaca` dataset will execute the following steps:

1. The raw text data will be processed into `input_ids` and `labels` using `AlpacaInstructDataHandler` and the `tokenizer` of `llama2_7b`.
2. `PackingHandler` will be used to perform packing on the processed `input_ids` and `labels`, resulting in concatenated `input_ids` and `labels` up to the `seq_length`. The `actual_seq_len` refers to the sequence length of each sub-sample in the concatenated sample. During training, this parameter will be used to generate the corresponding data mask.
3. If `compress_mask=False` is set in `adaptor_config`, a complete data mask will be returned during training. Otherwise, `actual_seq_len` will be returned.

#### Offline Dataset Processing

In addition to supporting online dataset loading and processing, `CommonDataLoader` also supports offline dataset processing and saving.

The [datasets_preprocess.py](https://gitee.com/mindspore/mindformers/blob/r1.5.0/toolkit/data_preprocess/huggingface/datasets_preprocess.py) script can be used to process Huggingface datasets offline and save them.

- Parameter Description

  | Parameter Name | Description                                                                                                                                                               | Type |
  |----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----:|
  | config         | Configuration file for offline data processing, which is used in the same way as online processing. Refer to [dataloader](#dataloader-parameter-description) for details. | str  |
  | save_path      | Path where the preprocessed dataset will be saved.                                                                                                                        | str  |
  | register_path  | Registration path for the model API, which includes the Python files related to the model, typically the model folder under the `research` directory.                     | int  |

- Usage Example

  You can use the configuration file provided in the [dataset packing](#dataset-packing) example and execute the following command.

  ```shell
  python toolkit/data_preprocess/huggingface/datasets_preprocess.py \
    --config data_process.yaml \
    --save_path /path/processed_data
  ```

  If you need to load the saved dataset, you should modify the YAML configuration as follows:

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
