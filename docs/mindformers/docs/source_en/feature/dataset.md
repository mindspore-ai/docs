# Datasets

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.10.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.10.0/docs/mindformers/docs/source_en/feature/dataset.md)

Currently, the MindSpore Transformers dynamic graph (PyNative) mode supports multiple dataset loading modes, covering common open-source and user-defined scenarios. The details are as follows:

- **Megatron dataset**: Datasets in the Megatron-LM format can be loaded, which is applicable to pretraining tasks of large-scale language models.
- **Hugging Face dataset**: Compatible with the Hugging Face `datasets` library, facilitating direct access to a wide range of public data resources in the community.
- **MindRecord dataset**: MindRecord is an efficient data storage and reading module provided by MindSpore. It can convert different public datasets into the MindRecord format for training.

## Configuration Structure

In dynamic graph mode, a dataset configuration is located in the `train_dataset` field of the YAML configuration file. The overall structure is as follows:

```yaml
train_dataset:
  dataloader:
    type: BlendedMegatronDatasetDataLoader  # Or HFDataLoader/MindDataset
    # ... Configuration items specific to each dataloader type
    column_names: ["input_ids", "labels"]   # Column names returned by the dataset
    shuffle: false                          # Whether to randomly shuffle the dataset
    python_multiprocessing: false           # Whether to use Python multi-process

  drop_remainder: true          # Whether to drop the last incomplete batch
  num_parallel_workers: 8       # Number of parallel worker threads for data loading
  prefetch_size: 1              # Number of prefetched batches
  numa_enable: false            # Whether to enable NUMA-aware data loading
```

The fields are described as follows.

| Parameter                               | Data Type| Required/Optional|            Default Value           | Value Description                                                                        |
|-------------------------------------|:----:|:----:|:-------------------------:|------------------------------------------------------------------------------|
| `dataloader.type`                   | str  |  Required |             -             | Data loader type. The value can be `BlendedMegatronDatasetDataLoader`, `HFDataLoader`, or `MindDataset`.|
| `dataloader.column_names`           | list |  Optional | `["input_ids", "labels"]` | List of column names returned by the dataset.                                                                  |
| `dataloader.shuffle`                | bool |  Optional |          `false`          | Specifies whether to randomly shuffle the dataset.                                                                |
| `dataloader.python_multiprocessing` | bool |  Optional |          `false`          | Specifies whether to use Python multi-process.                                                             |
| `drop_remainder`                    | bool |  Optional |          `true`           | Specifies whether to drop the last incomplete batch.                                                          |
| `num_parallel_workers`              | int  |  Optional |            `8`            | Number of parallel worker threads for data loading.                                                                |
| `prefetch_size`                     | int  |  Optional |            `1`            | Number of prefetched batches.                                                                |
| `numa_enable`                       | bool |  Optional |          `false`          | Specifies whether to enable NUMA-aware data loading.                                                           |

## Megatron Datasets

The Megatron dataset is an efficient data format designed for large-scale distributed language model pretraining. It is widely used in the Megatron-LM framework. Such dataset is usually preprocessed and serialized into a binary format (such as `.bin` or `.idx` files), and is accompanied by a specific indexing mechanism to facilitate efficient parallel loading and data splitting in a distributed cluster environment.

The following describes how to generate `.bin` or `.idx` files and how to use a Megatron dataset in training tasks.

### Data Preprocessing

MindSpore Transformers provides the data preprocessing script [preprocess_indexed_dataset.py](https://atomgit.com/mindspore/mindformers/blob/r2.0.0/toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py) to convert the original text corpus in `json` format into `.bin` or `.idx` files. If the original text is not in `json` format, you need to convert the data into the corresponding format.

The following is an example of a file in `json` format:

```json
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
```

The description of each data field is as follows.

| Field  | Description         | Required|
|-------|-------------|:------:|
| text  | Original text data.     |   Yes   |
| id    | Data ID, which is arranged in sequence.|   No   |
| src   | Data source.       |   No   |
| type  | Language type of the data.    |   No   |
| title | Data title.       |   No   |

The following uses the `wikitext-103` dataset as an example to describe how to convert a dataset into a Megatron dataset.

1. Download the `wikitext-103` dataset by going to [Link](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens).

2. Generate a data file in `json` format.

   The original text of the `wikitext-103` dataset is as follows:

   ```text
   = Valkyria Chronicles III =

   Valkyria Chronicles III is a tactical role-playing game developed by Sega for the PlayStation Portable.

   The game was released in Japan on January 27, 2011.

   = Gameplay =

   The game is similar to its predecessors in terms of gameplay...
   ```

   You need to process the original text into the following format and save it as a `json` file.

   ```json
   {"id": 0, "text": "Valkyria Chronicles III is a tactical role-playing game..."}
   {"id": 1, "text": "The game is similar to its predecessors in terms of gameplay..."}
   ```

3. Download the vocabulary file of the model.

   Different models correspond to different vocabulary files. Therefore, you need to download the vocabulary file corresponding to the training model. The `Qwen3-8B` model is used as an example. Download the [tokenizer](https://huggingface.co/Qwen/Qwen3-8B) for data preprocessing.

4. Generate a `.bin` or `.idx` data file.

   Run the data preprocessing script [preprocess_indexed_dataset.py](https://atomgit.com/mindspore/mindformers/blob/r2.0.0/toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py) to convert the original text data into the corresponding token IDs using the tokenizer of the model.

   The script parameters are as follows.

   | Parameter              | Description                                                                           |
   |-------------------|-------------------------------------------------------------------------------|
   | input             | Path of the `json` file.                                                                |
   | output-prefix     | Prefix of the `.bin` or `.idx` data file.                                                    |
   | tokenizer-type    | Type of the tokenizer used by the model.                                                           |
   | vocab-file        | Path of the tokenizer file (**tokenizer.model/vocab.json**) used by the model.                             |
   | merges-file       | Path of the tokenizer file (**merge.txt**) used by the model.                                              |
   | tokenizer-file    | Path of the tokenizer file (**tokenizer.json**) used by the model.                                         |
   | add_bos_token     | Specifies whether to add `bos_token` at the beginning of a sentence.                                                         |
   | add_eos_token     | Specifies whether to add `eos_token` at the end of a sentence.                                                         |
   | eos_token         | Token representing `eos_token`. The default value is `'</s>'`.                                              |
   | append-eod        | Specifies whether to add an `eos_token` at the end of the text.                                                     |
   | tokenizer-dir     | Directory of the HuggingFaceTokenizer used by the model. This parameter is valid only when `tokenizer-type` is set to **'HuggingFaceTokenizer'**.|
   | trust-remote-code | Specifies whether to allow the use of the tokenizer class defined on the Hub. This parameter is valid only when `tokenizer-type` is set to **'HuggingFaceTokenizer'**.   |
   | register_path     | Directory where the external tokenizer code is located. This parameter is valid only when `tokenizer-type` is set to **'AutoRegister'**.                 |
   | auto_register     | Import path of the external tokenizer. This parameter is valid only when `tokenizer-type` is set to **'AutoRegister'**.                  |

   The value of `tokenizer-type` can be `'HuggingFaceTokenizer'` or `'AutoRegister'`. If this parameter is set to `'HuggingFaceTokenizer'`, the AutoTokenizer class of the transformers library uses the tokenizer in the local Hugging Face repository for instantiation. If this parameter is set to `'AutoRegister'`, the external tokenizer class specified by the `register_path` and `auto_register` parameters is called.

   The [LlamaTokenizerFast](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/tokenizer_config.json) and [vocabulary](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/tokenizer.json) in the [Deepseek-V3 repository](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) are used as examples. If the corresponding repository does not exist on the localhost, manually download the configuration file (tokenizer_config.json) and vocabulary file (tokenizer.json) to a local directory, for example, `/path/to/huggingface/tokenizer`. Run the following command to process the dataset:

   ```shell
   python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
     --input /path/data.json \
     --output-prefix /path/megatron_data \
     --tokenizer-type HuggingFaceTokenizer \
     --tokenizer-dir /path/to/huggingface/tokenizer
   ```

### Model Pretraining

For MindSpore Transformers, it is recommended that you use Megatron datasets for model pretraining. You can generate a pretraining dataset by referring to [Data Preprocessing](#data-preprocessing). The following describes how to use a Megatron dataset in the configuration file in dynamic graph mode.

Modify the `train_dataset` part in the model configuration file as follows:

```yaml
train_dataset:
  dataloader:
    type: BlendedMegatronDatasetDataLoader
    datasets_type: GPTDataset
    sizes:
      - 8000   # Number of training set data samples
      - 0      # Number of test set data samples, which cannot be configured currently.
      - 0      # Number of evaluation set data samples, which cannot be configured currently.
    config:
      seed: 1234                         # Random seed for data sampling
      split: "1, 0, 0"                   # Ratio of the used training, test, and evaluation sets, which cannot be configured currently.
      seq_length: 8192                   # Sequence length of the data returned by the dataset
      eod_mask_loss: false               # Whether to compute the loss at the eod
      reset_position_ids: false          # Whether to reset **position_ids** at the eod
      create_attention_mask: false       # Whether to return **attention_mask**
      reset_attention_mask: falsee        # Whether to reset the **attention_mask** at the eod and return the ladder-like **attention_mask**
      create_compressed_eod_mask: false  # Whether to return the compressed **attention_mask**
      eod_pad_length: 128                # Length of **attention_mask** after compression.
      eod: 1                             # Token ID of the eod in the dataset
      pad: -1                            # Token ID of the pad in the dataset
      data_path:                         # Sampling ratio and path of the Megatron dataset
        - "1"                            # Dataset ratio
        - "/path/megatron_data"          # Path of the dataset **bin** file (excluding the .bin suffix)

    column_names: ["input_ids", "labels", "loss_mask", "position_ids"]
    shuffle: false
    python_multiprocessing: false

  drop_remainder: true
  num_parallel_workers: 8
  prefetch_size: 1
  numa_enable: false
```

The configuration items of `BlendedMegatronDatasetDataLoader` are described as follows.

| Parameter                               | Data Type| Required/Optional|     Default Value    | Value Description                                                                                |
|-------------------------------------|:----:|:----:|:-----------:|--------------------------------------------------------------------------------------|
| `datasets_type`                     | str  |  Required |      -      | Type of the Megatron dataset. Currently, only `GPTDataset` is supported.                                                   |
| `sizes`                             | list |  Required |      -      | A list of three elements, indicating the number of samples in the training set, test set, and evaluation set, respectively. Currently, only the training set is valid.                                              |
| `config.seed`                       | int  |  Optional |   `1234`    | Random seed for dataset sampling. The Megatron dataset randomly samples and concatenates samples based on this value.                                            |
| `config.split`                      | str  |  Optional | `"1, 0, 0"` | Ratio of the used training set, test set, and evaluation set, separated by commas (,). Currently, this parameter cannot be configured.                                                         |
| `config.seq_length`                 | int  |  Required |      -      | Sequence length of the data returned by the dataset, which must be the same as the sequence length of the training model.                                                          |
| `config.eod_mask_loss`              | bool |  Optional |   `false`   | Specifies whether to compute the loss at the end of the eod.                                                                    |
| `config.reset_position_ids`         | bool |  Optional |   `false`   | Specifies whether to reset **position_ids** at the end of the eod.                                                            |
| `config.create_attention_mask`      | bool |  Optional |   `true`    | Specifies whether to return **attention_mask**.                                                                 |
| `config.reset_attention_mask`       | bool |  Optional |   `false`   | Specifies whether to reset the **attention_mask** at the end of the eod and return a ladder-like **attention_mask**. This parameter is valid only under the condition of `create_attention_mask=true`. |
| `config.create_compressed_eod_mask` | bool |  Optional |   `false`   | Specifies whether to return the compressed **attention_mask** (that is, `actual_seq_len`). This parameter has a higher priority than `create_attention_mask`.           |
| `config.eod_pad_length`             | int  |  Optional |    `128`    | Length of the compressed **attention_mask**. This parameter is valid only under the condition of `create_compressed_eod_mask=true`.                   |
| `config.eod`                        | int  |  Required |      -      | Token ID of eod in the dataset.                                                                 |
| `config.pad`                        | int  |  Required |      -      | Token ID of pad in the dataset.                                                                 |
| `config.data_path`                  | list |  Required |      -      | List. Every two consecutive elements (a number and a string) in the list are regarded as a dataset, which indicates the sampling ratio of the dataset and the path of the dataset bin file without the suffix `.bin`. The sum of the ratios of all datasets must be 1.|

In addition, `column_names` needs to be adjusted based on the configurations of `create_attention_mask` and `create_compressed_eod_mask`.

- When `create_compressed_eod_mask=true`:

  ```yaml
  column_names: ["input_ids", "labels", "loss_mask", "position_ids", "actual_seq_len"]
  ```

- When `create_compressed_eod_mask=false` and `create_attention_mask=true`:

  ```yaml
  column_names: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
  ```

- When `create_compressed_eod_mask=false` and `create_attention_mask=false`:

  ```yaml
  column_names: ["input_ids", "labels", "loss_mask", "position_ids"]
  ```

After modifying the dataset-related configuration items in the model configuration file, you can start a model pretraining task by referring to the model document.

## Hugging Face Datasets

MindSpore Transformers interconnects with the [Hugging Face Datasets](https://huggingface.co/datasets) module (HF datasets for short), providing efficient and flexible loading and processing of HF datasets. The main features include:

1. **Diversified data loading**: Multiple data formats and loading modes of the HF `datasets` library are supported, easily adapting to data from different sources and structures.
2. **Abundant data processing APIs**: Compatible with multiple data processing methods (such as `sort`, `flatten`, and `shuffle`) of the `datasets` library, meeting common preprocessing requirements.
3. **Scalable data operations**: Users can customize dataset processing logic and use the efficient data **packing function**, which is suitable for optimization in large-scale training scenarios.

> To use HF datasets in MindSpore Transformers, you need to understand the basic functions of the `datasets` third-party library, such as dataset loading and processing. For details, see [link](https://huggingface.co/docs/datasets/loading).
>
> If the Python version is earlier than 3.10, install a version earlier than aiohttp 3.8.1.

### Configuration Description

To use the HF dataset functions in a model training task in dynamic graph mode, modify the `train_dataset` configurations in the YAML file.

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

> The parameters such as `seq_length` and `tokenizer` in all examples are obtained from the `Qwen3` model.

Parameters in `dataloader` are described as follows.

| Parameter                        | Data Type| Required/Optional|      Default Value      | Value Description                                                                     |
|------------------------------|:----:|:----:|:--------------:|---------------------------------------------------------------------------|
| `type`                       | str  |  Required |       -        | The value is fixed to `HFDataLoader`. This module supports the dataset loading and processing functions of the Hugging Face open-source community.                     |
| `load_func`                  | str  |  Optional | `load_dataset` | Specifies the API for loading a dataset. The options are `load_dataset` and `load_from_disk`. For details, see [Dataset Loading](#dataset-loading).|
| `create_attention_mask`      | bool |  Optional |    `false`     | Specifies whether to return the corresponding attention mask during dataset iteration.                                          |
| `create_compressed_eod_mask` | bool |  Optional |    `false`     | Specifies whether to return the compressed one-dimensional attention mask (`actual_seq_len`) during dataset iteration.                  |
| `compressed_eod_mask_length` | int  |  Optional |     `128`      | Length of the generated compressed attention mask. Generally, the value is the maximum number of EOD tokens in each sample in a dataset.                     |
| `shuffle`                    | bool |  Optional |    `false`     | Specifies whether to perform random sampling on a dataset.                                                             |
| `handler`                    | list |  Optional |       -        | Data preprocessing operations. For details, see [Dataset Processing](#dataset-processing).                                         |

### Dataset Loading

The dataset loading function is implemented by using the `load_func` parameter. `HFDataLoader` uses all parameters except those described in [Configuration Description](#configuration-description) as the input parameters of the dataset loading API. The usage is described as follows:

1. Use the `datasets.load_dataset` API to load a dataset.

   Set `load_func: 'load_dataset'` in the dataset configurations and the following parameters:

   1. **path (str)**—Path or name of the dataset folder.

    - If **path** is a local directory, the dataset is loaded from the supported files (such as CSV, JSON, and Parquet) in the directory, for example, `'/path/json/'`.
    - If **path** is the name of a dataset builder and **data_files** or **data_dir** is specified (available builders include "json", "csv", "parquet", "arrow", etc.), the dataset is loaded from the files in **data_files** or **data_dir**.

   2. **data_dir (str, optional)**—Dataset folder path, which is specified when **path** is set to the name of a dataset builder.

   3. **data_files (str, optional)**—Dataset file path, which is specified when **path** is set to the name of a dataset builder. It can be a single file or a list of file paths.

   4. **split (str)**—Split of the data to load. If this parameter is set to **None**, a dictionary containing all splits (usually datasets.Split.TRAIN and datasets.Split.TEST) is returned. If this parameter is specified, the corresponding split dataset instance is returned.

2. Use the `datasets.load_from_disk` API to load a dataset.

   Set `load_func: 'load_from_disk'` in the dataset configurations and the following parameters:

   - **dataset_path (str)**—Dataset folder path. This API is usually used to load the dataset saved using `datasets.save_to_disk`.

### Streaming Dataset Loading

When a dataset with a large number of samples is used, the device memory may be insufficient. In this case, you can use streaming loading to reduce the memory load. For details about the principle and related description of this function, see [stream](https://huggingface.co/docs/datasets/v4.0.0/en/stream).

To enable the streaming dataset loading function, add the following configurations to `dataloader` in [Configuration Description](#configuration-description):

```yaml
train_dataset:
  dataloader:
    type: HFDataLoader
    streaming: true
    size: 2000
    dataset_state_dir: '/path/dataset_state_dir'
    # ... Other configurations
```

Parameters

| Parameter                | Data Type| Required/Optional|   Default Value  | Value Description                                                                                                                                                                                                  |
|----------------------|:----:|:----:|:-------:|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `streaming`          | bool |  Optional | `false` | Specifies whether to enable the streaming dataset loading function.                                                                                                                                                                                         |
| `size`               | int  |  Optional |    -    | Specifies the total number of samples in a dataset iteration. When a dataset is loaded in streaming mode, an [IterableDataset](https://huggingface.co/docs/datasets/v4.0.0/en/package_reference/main_classes#datasets.IterableDataset) instance is created. The total number of samples cannot be obtained when all data is iterated. Therefore, this parameter needs to be specified.                  |
| `dataset_state_dir`  | str  |  Optional |    -    | Specifies the folder for saving and loading the dataset status. It is mainly used to save the dataset status when saving the weight and load the dataset status for resumable training.<br>The data offloading function is enabled for MindSpore datasets by default. Therefore, the dataset status is saved before the weight is saved.<br>When resumable training is performed by loading datasets in streaming mode, modifying parameters that affect `global_batch_size` (such as `data_parallel` and `batch_size`) will cause resumable training to fail and resampling to be performed for training.|

Currently, the streaming loading function has been verified in the following preprocessing scenarios:

1. Preprocessing the Alpaca dataset. The related configuration is `AlpacaInstructDataHandler`.
2. Preprocessing the Packing dataset. The related configuration is `PackingHandler`.
3. Renaming columns. The related configuration is `rename_column`.
4. Removing columns. The related configuration is `remove_columns`.

### Dataset Processing

`HFDataLoader` supports datasets native data processing and user-defined processing operations. Data preprocessing is implemented through the `handler` mechanism. This module preprocesses data based on the configuration sequence.

#### Native Data Processing Function

To rename or remove data columns, or randomly sample datasets, perform the following configurations:

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

1. **rename_column**: Renames a data column.

   In the example, `col1` can be renamed to `col2`.

2. **remove_columns**: Removes data columns.

   In the example, the renamed `col2` can be removed.

3. **shuffle**: Randomly shuffles datasets.

   In the example, 42 is used as the random seed to randomly sample the dataset.

For details about other native data processing of datasets, see the document [Datasets Process](https://huggingface.co/docs/datasets/process).

#### User-defined Data Processing

The user-defined data preprocessing function requires users to implement the data processing module. The following describes how to implement the user-defined data processing module. For details, see [AlpacaInstructDataHandler](https://atomgit.com/mindspore/mindformers/blob/r2.0.0/mindformers/dataset/handler/alpaca_handler.py).

User-defined data processing supports the following two formats: `Class` and `Method`.

If you use `Class` to construct the data processing module:

1. Implement the `Class` that contains the `__call__` function.

   ```python
   class CustomHandler:
       def __init__(self, seed):
           self.seed = seed

       def __call__(self, dataset):
           dataset = dataset.shuffle(seed=self.seed)
           return dataset
   ```

   The preceding `CustomHandler` implements random sampling of the dataset. To implement other functions, you can modify the data preprocessing operation and return the processed dataset.

   In addition, MindSpore Transformers provides [BaseInstructDataHandler](https://atomgit.com/mindspore/mindformers/blob/r2.0.0/mindformers/dataset/handler/base_handler.py) and has the built-in tokenizer configuration function. If you need to use the tokenizer, you can use the one that inherits the `BaseInstructDataHandler` class.

2. Add the call in [\_\_init\_\_.py](https://atomgit.com/mindspore/mindformers/blob/r2.0.0/mindformers/dataset/handler/__init__.py).

   ```python
   from .custom_handler import CustomHandler
   ```

3. Use `CustomHandler` in the configuration.

   ```yaml
   handler:
     - type: CustomHandler
       seed: 42
   ```

If you use `Method` to construct the data processing module:

1. Implement a function that contains the input parameters of the dataset instance.

   ```python
   def custom_process(dataset, seed):
       dataset = dataset.shuffle(seed)
       return dataset
   ```

2. Add the call in [\_\_init\_\_.py](https://atomgit.com/mindspore/mindformers/blob/r2.0.0/mindformers/dataset/handler/__init__.py).

   ```python
   from .custom_handler import custom_process
   ```

3. Use `custom_process` in the configuration.

   ```yaml
   handler:
     - type: custom_process
       seed: 42
   ```

### Application Examples

This section uses the `Qwen3` model and `alpaca` dataset as examples to describe how to fine-tune an HF dataset. `AlpacaInstructDataHandler` is required to process data online. The parameters are described as follows:

- `seq_length`: maximum length of the text encoded into token IDs by the tokenizer. Generally, it is the same as the sequence length used for model training.
- `padding`: specifies whether to pad token IDs to the maximum length during tokenizer encoding.
- `tokenizer`: `pretrained_model_dir` indicates the model vocabulary and weight folder downloaded from the HF community. `trust_remote_code` is usually set to `true`, and `padding_side` indicates that padding is performed on the right of the token ID.

#### Alpaca Dataset Fine-Tuning

The following uses fine-tuning of the `Qwen3` model as an example to describe how to modify the `Qwen3` model training configuration file.

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

After modifying the configuration file, you can start a fine-tuning task by referring to the `Qwen3` model document.

#### Packing Fine-Tuning of an Alpaca Dataset

MindSpore Transformers implements the packing function of datasets. It is mainly used to concatenate multiple short sequences into a fixed-length long sequence in foundation model training tasks to improve training efficiency. Currently, two strategies are supported, which can be configured using `pack_strategy`.

1. **pack**: Multiple samples are concatenated into a fixed-length sequence. If the length of a sample to be concatenated exceeds the maximum length specified by `seq_length`, the sample is placed in the next sample to be concatenated.
2. **truncate**: Multiple samples are concatenated into a fixed-length sequence. If the length of a sample to be concatenated exceeds the maximum length specified by `seq_length`, the sample is truncated and the remaining part is placed in the next sample to be concatenated.

This function is implemented using the `PackingHandler` class. The final output contains only the `input_ids`, `labels`, and `actual_seq_len` fields.

The following uses fine-tuning of the `Qwen3` model as an example to describe how to modify the `Qwen3` model training configuration file.

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

After modifying the configuration file, you can start a fine-tuning task by referring to the `Qwen3` model document.

## MindRecord Datasets

MindRecord is an efficient data storage and reading module provided by MindSpore. It reduces disk I/O and network I/O overheads, thereby providing a better data loading experience. For more details about its functions, see the [documentation](https://www.mindspore.cn/docs/en/r2.10.0/api_python/mindspore.mindrecord.html). This section describes how to use MindRecord in a dynamic graph training task of MindSpore Transformers.

The following uses `qwen3-8b` as an example to describe related functions. The script in the example applies only to the specified dataset. If you need to process a user-defined dataset, preprocess the data by referring to [MindRecord Format Conversion](https://www.mindspore.cn/tutorials/en/r2.10.0/dataset/record.html).

### Data Preprocessing

1. Download the `alpaca` dataset from [link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json).

2. Run the data processing script [alpaca_converter.py](https://atomgit.com/mindspore/docs/blob/r2.10.0/docs/mindformers/docs/source_zh_cn/static_graph/example/qwen3/alpaca_converter.py) to convert the `alpaca` dataset into a dialog format.

   ```shell
   python alpaca_converter.py \
     --data_path /path/alpaca_data.json \
     --output_path /path/alpaca-data-messages.json
   ```

   In the preceding information, `data_path` indicates the path of the downloaded `alpaca` dataset, and `output_path` indicates the path for storing the generated dialog-form data file.

3. Run the [datasets_preprocess.py](https://atomgit.com/mindspore/docs/blob/r2.10.0/docs/mindformers/docs/source_zh_cn/static_graph/example/qwen3/datasets_preprocess.py) script to convert the dialog-form data file into the MindRecord format.

   ```shell
   python datasets_preprocess.py \
     --input_glob /path/alpaca-data-messages.json \
     --tokenizer_dir /path/Qwen3-8B \
     --seq_length 32768 \
     --output_file /path/alpaca-messages.mindrecord
   ```

   The parameters in the script are described as follows:

   - `input_glob`: path for generating the dialog-form data file.
   - `tokenizer_dir`: path of the Qwen3 file.
   - `seq_length`: sequence length of the generated MindRecord data.
   - `output_file`: path for storing the generated MindRecord data.

### Model Fine-Tuning

You can generate a MindRecord dataset for `qwen3-8b` model fine-tuning by referring to the preceding data preprocessing process. The following describes how to use the generated data file to start a model fine-tuning task.

1. Modify the model configuration file.

   The `finetune_qwen3.yaml` configuration file is used for fine-tuning the `qwen3-8b` model. Modify the dataset configuration in the file as follows:

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

   To use the MindRecord dataset in a model training task, modify the following configuration items in the `dataloader` file:

   - `type`: data_loader type. Set this parameter to `MindDataset` when a MindRecord dataset is used.
   - `dataset_files`: path of the MindRecord data file. It can be the path of a single `.mindrecord` file, a list containing multiple file paths, or a directory containing `.mindrecord` files.
   - `shuffle`: specifies whether to randomly sample data samples during training.

2. Start model fine-tuning.

   After modifying the dataset-related configuration items in the model configuration file, you can start a model fine-tuning task by referring to the model document. The following uses the [Qwen3 model document](https://atomgit.com/mindspore/mindformers/blob/r2.0.0/configs/qwen3/README.md) as an example.

### Multi-Source Datasets

The native dataset loading module [MindDataset](https://www.mindspore.cn/docs/en/r2.10.0/api_python/dataset/mindspore.dataset.MindDataset.html) of the MindSpore framework has performance bottlenecks when loading and sampling multiple MindRecord datasets. Therefore, MindSpore Transformers uses `MultiSourceDataLoader` to efficiently load and sample multiple datasets.

The multi-source dataset function is enabled by modifying the `dataloader` configuration in the configuration file. The following is an example:

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

In the preceding information, the `shuffle` configuration affects the `shuffle_dataset` and `shuffle_file` parameters.

- `shuffle_dataset` indicates random sampling at the sub-dataset level.
- `shuffle_file` indicates random sampling at the sample level.

When different values are configured for `shuffle`, the following results are obtained.

| shuffle | shuffle_dataset | shuffle_file |
|---------|:---------------:|:------------:|
| true    |      true       |    true      |
| false   |      false      |    false     |
| infile  |      false      |     true     |
| files   |      true       |    false     |
| global  |      true       |     true     |

Other configuration items are described as follows.

| Parameter                   | Data Type | Required/Optional| Default Value| Value Description                                           |
|-------------------------|:-----:|:----:|:---:|-------------------------------------------------|
| `dataset_ratios`        | list  | Optional  | -   | Sampling ratio of each sub-dataset. The sum of sampling ratios of all sub-datasets is 1.                      |
| `samples_count`         |  int  |  Optional |  -  | Number of samples in each sub-dataset for sampling. This parameter is valid only when `dataset_ratios` is configured.      |
| `nums_per_dataset`      | list  |  Optional |  -  | Number of samples in each sub-dataset for sampling. This parameter is valid only when `dataset_ratios` is not configured.        |
| `sub_data_loader_args`  | dict  |  Optional |  -  | General configuration of each sub-dataset, which takes effect during the construction of all sub-datasets.                       |
| `sub_data_loader`       | list  |  Required |  -  | Configuration of each sub-dataset, which is the same as the `dataloader` configuration in a single MindRecord dataset.|
| `load_indices_npz_path` |  str  |  Optional |  -  | Path for loading the data index file.                                     |
| `save_indices_npz_path` |  str  |  Optional |  -  | Path for saving the data index file.                                     |
