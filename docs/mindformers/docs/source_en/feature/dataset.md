# Dataset

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/dataset.md)

MindSpore Transformers currently supports multiple types of dataset loading methods, covering common open-source and custom scenarios. Specifically, it includes:

- **Megatron Datasets**: Supports loading datasets in the Megatron-LM format, suitable for large-scale language model pre-training tasks.
- **HuggingFace Datasets**: Compatible with the HuggingFace datasets library, making it convenient to access a wide range of public data resources from the community.
- **MindRecord Datasets**: MindRecord is an efficient data storage and reading module provided by MindSpore. This module offers various methods to help users convert different public datasets into the MindRecord format, as well as tools for reading, writing, and retrieving data from MindRecord files.

## Megatron Dataset

Megatron dataset is an efficient data format designed for large-scale distributed language model pre-training scenarios, widely used within the Megatron-LM framework. These datasets are typically preprocessed and serialized into binary formats (such as `.bin` or `.idx` files), accompanied by specific indexing mechanisms to enable efficient parallel loading and data partitioning in distributed cluster environments.

The following sections will explain how to generate `.bin` and `.idx` files, as well as how to use Megatron datasets in training tasks.

### Data Preprocessing

MindSpore Transformers provides a data preprocessing script, [preprocess_indexed_dataset.py](https://gitee.com/mindspore/mindformers/blob/master/toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py), which is used to convert raw text data in `json` format into `.bin` and `.idx` files.

If the raw text data is not in `json` format, users need to preprocess and convert it into the appropriate format themselves.

Below is an example of a `json` format file:

```json
{"src": "www.nvidia.com", "text": "The quick brown fox", "type": "Eng", "id": "0", "title": "First Part"}
{"src": "The Internet", "text": "jumps over the lazy dog", "type": "Eng", "id": "42", "title": "Second Part"}
...
```

The descriptions for each data field are as follows:

| Field Name | Description                  | Required  |
|------------|------------------------------|:---------:|
| text       | Raw text data                |    Yes    |
| id         | Unique identifier (in order) |    No     |
| src        | Data source                  |    No     |
| type       | Language type                |    No     |
| title      | Data title                   |    No     |

The following example demonstrates how to convert the `wikitext-103` dataset into a Megatron dataset format:

1. Download the `wikitext-103` dataset: [Link](https://dagshub.com/DagsHub/WIkiText-103/src/main/dataset/tokens)

2. Generate a `json` format data file

   The original text of the `wikitext-103` dataset looks like this:

   ```text
   = Valkyria Chronicles III =

   Valkyria Chronicles III is a tactical role-playing game developed by Sega for the PlayStation Portable.

   The game was released in Japan on January 27, 2011.

   = Gameplay =

   The game is similar to its predecessors in terms of gameplay...
   ```

   You need to preprocess the original text into the following format and save it as a `json` file:

   ```json
   {"id": 0, "text": "Valkyria Chronicles III is a tactical role-playing game..."}
   {"id": 1, "text": "The game is similar to its predecessors in terms of gameplay..."}
   ...
   ```

3. Download the model's vocabulary file

   Since different models use different vocabulary files, you need to download the corresponding vocabulary file for the training model.
   Taking the `Llama3` model as an example, download the [tokenizer.model](https://huggingface.co/meta-llama/Meta-Llama-3-8B/blob/main/original/tokenizer.model) for data preprocessing.

4. Generate `.bin` and `.idx` data files

   Run the data preprocessing script [preprocess_indexed_dataset.py](https://gitee.com/mindspore/mindformers/blob/master/toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py) to convert the original text data into corresponding token IDs using the model's tokenizer.

   The script accepts the following parameters:

   | Parameter Name    | Description                                                                                                       |
   |-------------------|-------------------------------------------------------------------------------------------------------------------|
   | input             | Path to the `json` format file                                                                                    |
   | output-prefix     | Prefix for the `.bin` and `.idx` data files                                                                       |
   | tokenizer-type    | Type of tokenizer used by the model                                                                               |
   | vocab-file        | Path to the model’s tokenizer file (`tokenizer.model` / `vocab.json`)                                             |
   | merges-file       | Path to the model’s tokenizer merges file (`merge.txt`)                                                           |
   | tokenizer-file    | Path to the model’s tokenizer file (`tokenizer.json`)                                                             |
   | add_bos_token     | Whether to add a `bos_token` (beginning of sequence token) to the vocabulary                                      |
   | add_eos_token     | Whether to add an `eos_token` (end of sequence token) to the vocabulary                                           |
   | eos_token         | The string represent token `eos_token`, defaults to '</s>'                                                        |
   | append-eod        | Whether to add an `eos_token` to the end of documentation                                                         |
   | tokenizer-dir     | The directory of HuggingFaceTokenizer, Take effects only when `tokenizer-type`='HuggingFaceTokenizer'             |
   | trust-remote-code | Whether to allow for custom models defined in Hub. Take effects only when `tokenizer-type`='HuggingFaceTokenizer' |
   | register_path     | Set the code directory of outer tokenizer. Take effects only when `tokenizer-type`='AutoRegister'                 |
   | auto_register     | Set the import path of outer tokenizer. Take effects only when `tokenizer-type`='AutoRegister'                    |

   The optional value of `tokenizer-type` is 'HuggingFaceTokenizer' and 'AutoRegister'. When it's set to 'HuggingFaceTokenizer', `AutoTokenizer` class in `transformers` library will instantiate tokenizer in local HuggingFace repository. When it's set to 'AutoRegister', outer tokenizer class specified by `register_path` and `auto_register` will be applied.

   Take [LlamaTokenizerFast](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/tokenizer_config.json) and [vocab file](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/tokenizer.json) in [DeepSeek-V3 repository](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base) as an example. If there is no corresponding repository, configuration file (tokenizer_config.json) and vocab file (tokenizer.json) needed to be download to local path. Let it be /path/to/huggingface/tokenizer. Execute the following command to preprocess the dataset:

   ```shell
   python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
     --input /path/data.json \
     --output-prefix /path/megatron_data \
     --tokenizer-type HuggingFaceTokenizer \
     --tokenizer-dir /path/to/huggingface/tokenizer
   ```

   Take outer tokenizer class [Llama3Tokenizer](https://gitee.com/mindspore/mindformers/blob/master/research/llama3_1/llama3_1_tokenizer.py) as an example, make sure **local** MindSpore Transformers repository has 'research/llama3_1/llama3_1_tokenizer.py', and execute the following command to preprocess the dataset:

   ```shell
   python toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py \
     --input /path/data.json \
     --output-prefix /path/megatron_data \
     --tokenizer-type AutoRegister \
     --vocab-file /path/tokenizer.model \
     --register_path research/llama3_1 \
     --auto_register llama3_1_tokenizer.Llama3Tokenizer
   ```

### Model Pre-training

MindSpore Transformers recommends using Megatron datasets for model pre-training.
Based on the [Data Preprocessing](#data-preprocessing) steps, you can generate the required pre-training dataset.
The following explains how to configure and use Megatron datasets in the configuration file.

1. Prepare the `parallel_speed_up.json` file

   Megatron dataset relies on the `dataset_broadcast_opt_level` feature for data broadcasting.
   For more details, refer to the [documentation](https://www.mindspore.cn/docs/zh-CN/master/api_python/parallel/mindspore.parallel.auto_parallel.AutoParallel.html).
   Therefore, you need to create a `parallel_speed_up.json` file with the following content:

   ```json
   {
       "dataset_broadcast_opt_level": 3
   }
   ```

   At the same time, add the following fields to the model configuration file:

   ```yaml
   context:
     ascend_config:
       parallel_speed_up_json_path: "/path/to/parallel_speed_up.json"
   ```

2. Modify the model configuration file

   To use the Megatron dataset in model pre-training tasks, mainly modify the `train_dataset` section in the configuration file.

   ```yaml
    train_dataset: &train_dataset
      data_loader:
        type: BlendedMegatronDatasetDataLoader
        datasets_type: "GPTDataset"
        sizes:
          - 1000 # Number of training dataset samples
          - 0    # Number of testing dataset samples (currently unsupported)
          - 0    # Number of evaluation dataset samples (currently unsupported)
        config:  # GPTDataset configuration options
          seed: 1234                        # Random seed for data sampling
          split: "1, 0, 0"                  # Ratio of training, testing, and evaluation datasets (currently unsupported)
          seq_length: 8192                  # Sequence length of data returned by the dataset
          eod_mask_loss: True               # Whether to compute loss at end-of-document (EOD) tokens
          reset_position_ids: True          # Whether to reset position_ids at EOD tokens
          create_attention_mask: True       # Whether to return attention_mask
          reset_attention_mask: True        # Whether to reset attention_mask at EOD tokens, returning a staircase-shaped mask
          create_compressed_eod_mask: False # Whether to return a compressed attention_mask
          eod_pad_length: 128               # Length of the compressed attention_mask
          eod: 0                            # Token ID of the EOD token in the dataset
          pad: 1                            # Token ID of the pad token in the dataset

          data_path:                        # Sampling ratio and paths for Megatron datasets
            - '0.3'                         # Ratio of dataset1
            - "/path/megatron_data1"        # Path to bin file of dataset1 excluding the .bin suffix
            - '0.7'                         # Ratio of dataset2
            - "/path/megatron_data2"        # Path to bin file of dataset2 excluding the .bin suffix

      input_columns: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
      construct_args_key: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]

    parallel:
      full_batch: False
      dataset_strategy: [[*dp, 1], [*dp, 1], [*dp, 1], [*dp, 1], [*dp, 1, 1, 1]]  # *dp means same value as data_parallel

    model_config:
      input_sliced_sig: True
   ```

   Below are the descriptions for each configuration option of the `GPTDataset` in the dataset:

   | Parameter Name             | Description                                                                                                                                                                                                                            |
   |----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
   | seed                       | Random seed for dataset sampling. Megatron datasets use this value to randomly sample and concatenate samples. Default: `1234`                                                                                                         |
   | seq_length                 | Sequence length of data returned by the dataset. Should be consistent with the sequence length of the training model.                                                                                                                  |
   | eod_mask_loss              | Whether to compute loss at the end-of-document (EOD) token. Default: `False`                                                                                                                                                           |
   | create_attention_mask      | Whether to return an attention_mask. Default: `True`                                                                                                                                                                                   |
   | reset_attention_mask       | Whether to reset the attention_mask at EOD tokens, returning a staircase-shaped attention_mask. Effective only if `create_attention_mask=True`. Default: `False`                                                                       |
   | create_compressed_eod_mask | Whether to return a compressed attention_mask. Has higher priority than `create_attention_mask`. Default: `False`                                                                                                                      |
   | eod_pad_length             | Length of the compressed attention_mask. Effective only if `create_compressed_eod_mask=True`. Default: `128`                                                                                                                           |
   | eod                        | Token ID of the EOD token in the dataset                                                                                                                                                                                               |
   | pad                        | Token ID of the pad token in the dataset                                                                                                                                                                                               |
   | data_path                  | List, every two consecutive elements (number, string) are considered as a dataset, represent ratio of the dataset and the path to its bin file excluding `.bin` suffix respectively. The sum of datasets' ratios should be equal to 1. |

   In addition, the Megatron dataset also depends on configurations such as `input_columns`, `construct_args_key`, and `full_batch`. For more details, refer to the [configuration file documentation](https://www.mindspore.cn/mindformers/docs/zh-CN/master/feature/configuration.html).

   Here, we only explain how to configure them in different scenarios:

    - When `create_compressed_eod_mask=True`:

    ```yaml
    train_dataset: &train_dataset
      input_columns: ["input_ids", "labels", "loss_mask", "position_ids", "actual_seq_len"]
      construct_args_key: ["input_ids", "labels", "loss_mask", "position_ids", "actual_seq_len"]
    parallel:
      full_batch: False
      dataset_strategy: [[*dp, 1], [*dp, 1], [*dp, 1], [*dp, 1], [*dp, 1]]  # *dp means same value as data_parallel
    ```

    - When `create_compressed_eod_mask=False` and `create_attention_mask=True`:

    ```yaml
    train_dataset: &train_dataset
      input_columns: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
      construct_args_key: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
    parallel:
      full_batch: False
      dataset_strategy: [[*dp, 1], [*dp, 1], [*dp, 1], [*dp, 1], [*dp, 1, 1, 1]]  # *dp means same value as data_parallel
    ```

    - When `create_compressed_eod_mask=False` and `create_attention_mask=False`:

    ```yaml
    train_dataset: &train_dataset
      input_columns: ["input_ids", "labels", "loss_mask", "position_ids"]
      construct_args_key: ["input_ids", "labels", "loss_mask", "position_ids"]
    parallel:
      full_batch: False
      dataset_strategy: [[*dp, 1], [*dp, 1], [*dp, 1], [*dp, 1]]  # *dp means same value as data_parallel
    ```

3. Start Model Pre-training

   After modifying the dataset and parallel-related configurations in the model configuration file, you can refer to the model documentation to launch the model pre-training task.
   Here, we take the [Llama3_1 model documentation](https://gitee.com/mindspore/mindformers/blob/master/research/llama3_1/README.md) as an example.

## Hugging Face Dataset

The HuggingFace Dataset (HF Dataset) module is integrated with the [HuggingFace community](https://huggingface.co/datasets), providing efficient and flexible **HF dataset loading and processing**. Main features include:

1. **Diverse Data Loading**: Supports various formats and loading methods from the Hugging Face `datasets` library, easily adapting to different sources and structures.
2. **Rich Data Processing Interfaces**: Compatible with multiple processing methods from the `datasets` library (such as `sort`, `flatten`, `shuffle`, etc.), meeting common preprocessing needs.
3. **Extensible Data Operations**: Supports user-defined dataset processing logic and provides efficient **packing functionality** for large-scale training optimization.

> To use HuggingFace datasets in MindSpore Transformers, you need to understand the basic functionalities of the `datasets` third-party library, such as dataset loading and processing. For more details, please refer to [this link](https://huggingface.co/docs/datasets/loading).

### Configuration

To use HF dataset functionality in model training, modify the `data_loader` configuration:

```yaml
train_dataset: &train_dataset
  input_columns: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
  construct_args_key: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]

  data_loader:
    type: HFDataLoader

    # datasets load arguments
    load_func: 'load_dataset'
    path: "json"
    data_files: "/path/alpaca-gpt4-data.json"
    split: "train"

    # MindFormers dataset arguments
    create_attention_mask: True
    create_compressed_eod_mask: False
    compressed_eod_mask_length: 128
    use_broadcast_data: True
    shuffle: False

    # dataset process arguments
    handler:
      - type: AlpacaInstructDataHandler
        seq_length: 4096
        padding: False
        tokenizer:
          pretrained_model_dir: '/path/qwen3'
          trust_remote_code: True
          padding_side: 'right'
      - type: PackingHandler
        seq_length: 4096
        pack_strategy: 'pack'

  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  numa_enable: False
  prefetch_size: 1
  seed: 1234
```

> All examples use `seq_length`, `tokenizer`, etc., from the `qwen3` model.

`data_loader` parameter descriptions:

| Parameter                  | Description                                                                                                                                                                                                                                      | Type |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----:|
| type                       | Fixed as `HFDataLoader`. This module supports dataset loading and processing function from the HuggingFace open-source community and can also be configured as `CommonDataLoader`. However, this interface will be deprecated in future versions | str  |
| load_func                  | Specifies the dataset loading interface, options are `load_dataset` and `load_from_disk`. See [Dataset Loading](#dataset-loading). Default is `load_dataset`.                                                                                    | str  |
| create_attention_mask      | Whether to return attention mask during dataset iteration; default is `False`                                                                                                                                                                    | bool |
| create_compressed_eod_mask | Whether to return compressed one-dimensional attention mask during iteration; default is `False`                                                                                                                                                 | bool |
| compressed_eod_mask_length | Length of compressed attention mask, usually the max number of eod tokens in samples; default is `128`                                                                                                                                           | int  |
| use_broadcast_data         | Whether to enable data broadcast; default is `True`. Enabling this configuration can reduce memory and I/O overhead.                                                                                                                             | bool |
| shuffle                    | Whether to randomly sample the dataset; default is `False`                                                                                                                                                                                       | bool |
| handler                    | Data preprocessing operations. For details, refer to the [Dataset Processing](#dataset-processing) section                                                                                                                                       | list |

### Dataset Loading

The dataset loading functionality is mainly implemented through the `load_func` parameter.
`HFDataLoader` will pass all parameters (except those defined in [Configuration](#configuration)) as input arguments to the dataset loading interface. The detailed usage is as follows:

1. Using the `datasets.load_dataset` interface to load datasets:

   In the dataset configuration, set `load_func: 'load_dataset'`, and configure the following parameters:

    1. **path (str)** — Path or name of the dataset directory.

        - If `path` is a local directory, the dataset will be loaded from the supported files (csv, json, parquet, etc.) in that directory. Example: `'/path/json/'`.
        - If `path` is the name of a dataset builder and `data_files` or `data_dir` is specified (available builders include `"json"`, `"csv"`, `"parquet"`, `"arrow"`, etc.), the dataset will be loaded from the files in `data_files` or `data_dir`.

    2. **data\_dir (str, optional)** — When `path` is set to the name of a dataset builder, this specifies the dataset directory path.

    3. **data\_files (str, optional)** — When `path` is set to the name of a dataset builder, this specifies the dataset file path(s). It can be a single file or a list of multiple file paths.

    4. **split (str)** — The data split to load. If set to `None`, a dictionary containing all splits will be returned (typically `datasets.Split.TRAIN` and `datasets.Split.TEST`). If specified, the corresponding split will be returned as a `Dataset` instance.

2. Using the `datasets.load_from_disk` interface to load datasets:

   In the dataset configuration, set `load_func: 'load_from_disk'`, and configure the following parameter:

    - **dataset\_path (str)** — Path to the dataset directory. This interface is typically used to load datasets that have been preprocessed offline or saved using `datasets.save_to_disk`.

### Dataset Processing

`HFDataLoader` supports native datasets processing and user-defined operations, mainly via the `handler` mechanism, which executes preprocessing steps in order.

#### Native Processing

To rename dataset columns, remove columns, or randomly sample the dataset, you can configure as follows:

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

1. rename_column - Rename a column

   Renames `col1` to `col2`.

2. remove_columns - Remove a column

   Removes `col2`.

3. shuffle - Shuffle the dataset

   Shuffles with seed 42.

For other native dataset processing operations, please refer to the [datasets process](https://huggingface.co/docs/datasets/process) documentation.

#### Custom Processing

To use custom preprocessing, implement your own handler module. See [AlpacaInstructDataHandler](https://gitee.com/mindspore/mindformers/blob/master/mindformers/dataset/handler/alpaca_handler.py).

Custom handlers support `Class` and `Method` forms:

If using a `Class`:

1. Implement a class with a __call__ function:

   ```python
   class CustomHandler:
       def __init__(self, seed):
           self.seed = seed

       def __call__(self, dataset):
           dataset = dataset.shuffle(seed=self.seed)
           return dataset
   ```

   The `CustomHandler` above implements the random sampling of the dataset. To achieve other functions, you can modify the data preprocessing operations and return the processed dataset.

   MindFormers provides [BaseInstructDataHandler](https://gitee.com/mindspore/mindformers/blob/master/mindformers/dataset/handler/base_handler.py) with built-in tokenizer config. If need to use a tokenizer, you can inherit from the `BaseInstructDataHandler` class.

2. Add to [\_\_init__.py](https://gitee.com/mindspore/mindformers/blob/master/mindformers/dataset/handler/__init__.py):

   ```python
   from .custom_handler import CustomHandler
   ```

3. Use in config:

   ```yaml
   handler:
     - type: 'custom_process'
       seed: 42
   ```

If using a `Method`:

1. Implement a function with dataset as input:

   ```python
   def custom_process(dataset, seed):
       dataset = dataset.shuffle(seed)
       return dataset
   ```

2. Add to [\_\_init__.py](https://gitee.com/mindspore/mindformers/blob/master/mindformers/dataset/handler/__init__.py):

   ```python
   from .custom_handler import custom_process
   ```

3. Use in config:

   ```yaml
   handler:
     custom_process:
       seed: 42
   ```

### Practical Application

Below, we will use the `qwen3` model and the `alpaca` dataset as examples to demonstrate how to fine-tune the HF dataset. The `AlpacaInstructDataHandler` will be used for online data processing. The specific parameter descriptions are as follows.

- seq_length: Maximum length for encoding text to token IDs via tokenizer; usually matches model training sequence length.
- padding: Whether to pad token IDs to max length during encoding.
- tokenizer: `pretrained_model_dir` is the folder with model vocab and weights from HF. `trust_remote_code` is usually set to `True`, and `padding_side` indicates that padding is applied from the right side of the token ID.

#### Alpaca Dataset Fine-tuning

For `qwen3` model fine-tuning, modify the training config:

```yaml
train_dataset: &train_dataset
  input_columns: ["input_ids", "labels"]
  construct_args_key: ["input_ids", "labels"]

  data_loader:
    type: HFDataLoader

    # datasets load arguments
    load_func: 'load_dataset'
    path: 'json'
    data_files: '/path/alpaca-gpt4-data.json'

    # MindFormers dataset arguments
    use_broadcast_data: True
    shuffle: False

    # dataset process arguments
    handler:
      - type: AlpacaInstructDataHandler
        seq_length: 4096
        padding: True
        tokenizer:
          pretrained_model_dir: '/path/qwen3'  # qwen3 repo dir
          trust_remote_code: True
          padding_side: 'right'

  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  numa_enable: False
  prefetch_size: 1
  seed: 1234

context:
  ascend_config:
    parallel_speed_up_json_path: "configs/qwen3/parallel_speed_up.json"

parallel_config:
  data_parallel: &dp 2

parallel:
  full_batch: False
  dataset_strategy: [
    [*dp, 1],
    [*dp, 1]
  ]  # *dp = data_parallel
```

See [Megatron Dataset](#Megatron-dataset) for details on `parallel_speed_up_json_path`, `dataset_strategy`, etc.

After modifying the configuration file, refer to the `qwen3` model documentation to initiate a fine-tuning task that loads offline data.

#### Alpaca Dataset Packing Fine-tuning

MindSpore Transformers implements the dataset packing functionality, which is mainly used in large-scale model training tasks to concatenate multiple short sequences into fixed-length long sequences, thereby improving training efficiency. It currently supports two strategies, which can be configured through `pack_strategy`:

1. **pack**: Concatenates multiple samples into a fixed-length sequence. When the sample to be concatenated exceeds the maximum length `seq_length`, the sample is placed into the next concatenated sequence.
2. **truncate**: Concatenates multiple samples into a fixed-length sequence. When the sample to be concatenated exceeds the maximum length `seq_length`, the sample is truncated, and the remaining part is placed into the next concatenated sequence.

This functionality is implemented through the `PackingHandler` class. The final output only contains three fields: `input_ids`, `labels`, and `actual_seq_len`.

For packing fine-tuning with `qwen3`, modify the training config:

```yaml
train_dataset: &train_dataset
  input_columns: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
  construct_args_key: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]

  data_loader:
    type: HFDataLoader

    # datasets load arguments
    load_func: 'load_dataset'
    path: 'json'
    data_files: '/path/alpaca-gpt4-data.json'

    # MindFormers dataset arguments
    use_broadcast_data: True
    shuffle: False

    # dataset process arguments
    handler:
      - type: AlpacaInstructDataHandler
        seq_length: 4096
        padding: False
        tokenizer:
          pretrained_model_dir: '/path/qwen3'  # qwen3 repo dir
          trust_remote_code: True
          padding_side: 'right'
      - type: PackingHandler
        seq_length: 4096
        pack_strategy: 'pack'

  num_parallel_workers: 8
  python_multiprocessing: False
  drop_remainder: True
  numa_enable: False
  prefetch_size: 1
  seed: 1234

context:
  ascend_config:
    parallel_speed_up_json_path: "configs/qwen3/parallel_speed_up.json"

parallel_config:
  data_parallel: &dp 2

parallel:
  full_batch: False
  dataset_strategy: [
    [*dp, 1],
    [*dp, 1],
    [*dp, 1],
    [*dp, 1],
    [*dp, 1, 1, 1]
  ]  # *dp = data_parallel
```

After modifying the config, refer to the `qwen3` model documentation to start fine-tuning.

#### Offline Processing for Alpaca Data Fine-tuning

`HFDataLoader` supports offline processing and saving of HF datasets; processed data can be loaded directly for training.

1. Modify the `qwen3` training config:

   ```yaml
   train_dataset: &train_dataset
     data_loader:
       type: HFDataLoader

       # datasets load arguments
       load_func: 'load_dataset'
       path: 'json'
       data_files: '/path/alpaca-gpt4-data.json'

       # dataset process arguments
       handler:
         - type: AlpacaInstructDataHandler
           seq_length: 4096
           padding: False
           tokenizer:
             pretrained_model_dir: '/path/qwen3'  # qwen3 repo dir
             trust_remote_code: True
             padding_side: 'right'
         - type: PackingHandler
           seq_length: 4096
           pack_strategy: 'pack'
   ```

2. Run the preprocessing script:

   ```shell
   python toolkit/data_preprocess/huggingface/datasets_preprocess.py --config configs/qwen3/pretrain_qwen3_32b_4k.yaml --save_path processed_dataset/
   ```

3. Modify the config:

   ```yaml
   train_dataset: &train_dataset
     input_columns: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]
     construct_args_key: ["input_ids", "labels", "loss_mask", "position_ids", "attention_mask"]

     data_loader:
       type: HFDataLoader

       # datasets load arguments
       load_func: 'load_from_disk'
       dataset_path: '/path/processed_dataset'

       # MindFormers dataset arguments
       create_attention_mask: True
       use_broadcast_data: True
       shuffle: False

     num_parallel_workers: 8
     python_multiprocessing: False
     drop_remainder: True
     numa_enable: False
     prefetch_size: 1
     seed: 1234

   context:
     ascend_config:
       parallel_speed_up_json_path: "configs/qwen3/parallel_speed_up.json"

   parallel_config:
     data_parallel: &dp 2

   parallel:
     full_batch: False
     dataset_strategy: [
       [*dp, 1],
       [*dp, 1],
       [*dp, 1],
       [*dp, 1],
       [*dp, 1, 1, 1]
     ]  # *dp = data_parallel
   ```

   After modifying the configuration file, refer to the `qwen3` model documentation to initiate a fine-tuning task that loads offline data.

## MindRecord Dataset

MindRecord is an efficient data storage and reading module provided by MindSpore. It reduces disk IO and network IO overhead, resulting in a better data loading experience. For more detailed feature introductions, refer to the [documentation](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mindrecord.html). Here, we only cover how to use MindRecord in MindSpore Transformers model training tasks.

The following example uses `qwen2_5-0.5b` fine-tuning to explain related functionalities.

### Data Preprocessing

1. Download the `alpaca` dataset: [Link](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)

2. Execute the data processing script to convert the `alpaca` dataset into a dialogue format:

   ```shell
   python research/qwen2/alpaca_converter.py \
     --data_path /path/alpaca_data.json \
     --output_path /path/alpaca-data-messages.json
   ```

   Here, `data_path` refers to the path where the downloaded `alpaca` dataset is stored, and `output_path` refers to the save path for the generated dialogue format data file.

3. Execute the script to convert the dialogue format data file into MindRecord format:

   ```shell
   python research/qwen2/qwen2_preprocess.py \
     --dataset_type 'qa' \
     --input_glob /path/alpaca-data-messages.json \
     --vocab_file /path/vocab.json \
     --merges_file /path/merges.txt \
     --seq_length 32768 \
     --output_file /path/alpaca-messages.mindrecord
   ```

   The script parameters are explained as follows:

    - `dataset_type`: Type of data preprocessing. For the alpaca dataset, set this to `qa`.
    - `input_glob`: Path to the dialogue format data file.
    - `vocab_file`: Path to the `vocab.json` file of the qwen2 model.
    - `merges_file`: Path to the `merges.txt` file of the qwen2 model.
    - `seq_length`: Sequence length for generating MindRecord data.
    - `output_file`: Save path for the generated MindRecord data.

   > The `vocab_file` and `merges_file` can be obtained from the qwen2 model repository on the HuggingFace community.

### Model Fine-tuning

Following the above data preprocessing steps, you can generate a MindRecord dataset for fine-tuning the `qwen2_5-0.5b` model. Below is an introduction on how to use the generated data file to start the model fine-tuning task.

1. Modify the model configuration file

   The `qwen2_5-0.5b` model fine-tuning uses the [finetune_qwen2_5_0.5b_8k.yaml](https://gitee.com/mindspore/mindformers/blob/master/research/qwen2_5/finetune_qwen2_5_0_5b_8k.yaml) configuration file. Modify the dataset section as follows:

   ```yaml
   train_dataset: &train_dataset
     data_loader:
       type: MindDataset
       dataset_dir: "/path/alpaca-messages.mindrecord"
       shuffle: True
   ```

   When using the MindRecord dataset in a model training task, the following configurations in `data_loader` need to be modified:

    - `type`: Type of data_loader. Set to `MindDataset` when using MindRecord datasets.
    - `dataset_dir`: Path to the MindRecord data files.
    - `shuffle`: Whether to randomly sample data samples during training.

2. Start Model Fine-tuning

   After modifying the dataset and parallel-related configurations in the model configuration file, you can refer to the model documentation to launch the fine-tuning task. Here, we take the [Qwen2_5 model documentation](https://gitee.com/mindspore/mindformers/blob/master/research/qwen2_5/README.md) as an example.

### Multi-source Datasets

The native MindSpore dataset loading module [MindDataset](https://www.mindspore.cn/docs/zh-CN/master/api_python/dataset/mindspore.dataset.MindDataset.html) has performance bottlenecks when loading and sampling multiple MindRecord datasets.

Therefore, MindSpore Transformers implements the `MultiSourceDataLoader` to achieve efficient loading and sampling across multiple datasets.

The multi-source dataset functionality is mainly enabled by modifying the `data_loader` configuration in the config file. Below is an example:

```yaml
train_dataset: &train_dataset
  data_loader:
    type: MultiSourceDataLoader
    data_source_type: random_access
    shuffle: True
    dataset_ratios: [0.2, 0.8]
    samples_count: 1000
    nums_per_dataset: [2000]
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
```

The `shuffle` setting affects two parameters: `shuffle_dataset` and `shuffle_file`:

- `shuffle_dataset` indicates random sampling at the sub-dataset level.
- `shuffle_file` indicates random sampling at the sample level.

The effects of different `shuffle` values are as follows:

| shuffle |  shuffle_dataset  |  shuffle_file  |
|---------|:-----------------:|:--------------:|
| True    |       True        |      True      |
| False   |       False       |     False      |
| infile  |       False       |      True      |
| files   |       True        |     False      |
| global  |       True        |      True      |

Other configuration parameters are explained below:

| Parameter             | Description                                                                                   | Type |
|-----------------------|-----------------------------------------------------------------------------------------------|:----:|
| dataset_ratios        | Sampling ratios for each sub-dataset; sum of all equals 1                                     | list |
| samples_count         | Number of samples from each sub-dataset, effective only when `dataset_ratios` is configured   | int  |
| nums_per_dataset      | Number of samples per sub-dataset, effective when `dataset_ratios` is not configured          | list |
| sub_data_loader_args  | Common configurations for each sub-dataset, effective during sub-dataset construction         | dict |
| sub_data_loader       | Configuration for each sub-dataset, same as `data_loader` config in single MindRecord dataset | list |
| load_indices_npz_path | Path to load data index file                                                                  | str  |
| save_indices_npz_path | Path to save data index file                                                                  | str  |
