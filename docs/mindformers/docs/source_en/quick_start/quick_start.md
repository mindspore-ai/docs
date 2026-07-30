# Quick Start

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.10.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.10.0/docs/mindformers/docs/source_en/quick_start/quick_start.md)

## Overview

This section helps you quickly use dynamic graphs (PyNative) to complete an LLM training and gain an intuitive understanding of the overall process. The following uses **DeepSeek-V3** as an example to perform the minimum-scale pre-training on two devices.

Dynamic graph training uses the unified script `run_mindformer.py` as the entry point and routes to the dynamic graph trainer `mindformers.pynative.trainer.Trainer` through `--mode 1`. The trainer builds the model, dataset, and optimizer and drives the training loop. A task can be summarized into three steps:

1. **Preparing the configuration file**: Compile a dynamic graph YAML file to connect the model, data, parallelism, and optimizer.
2. **Starting training**: Use `msrun` to start multi-device training.
3. **Viewing the result**: Check the logs to ensure that the loss/grad norm decreases properly in each step.

For details about the complete process and detailed configuration of each capability, see [Function Overview](../feature/overview.md). (The training guide and feature pages will be released later.)

## Prerequisites

- MindSpore and MindSpore Transformers have been installed. For details, see [Installation Guide](../installation.md).
- The Ascend hardware environment has been set up, and the CANN has been correctly configured.
- A **Megatron dataset** (`.bin`/`.idx`) has been prepared. The repository script [`preprocess_indexed_dataset.py`](https://atomgit.com/mindspore/mindformers/blob/r2.0.0/toolkit/data_preprocess/megatron/preprocess_indexed_dataset.py) can be used to create a dataset (JSON is transmitted to BIN/IDX). For details, see "Dataset."

> **Configuring the Data Path Connection**
>
> The preprocessing output is a pair of `xxx.bin` and `xxx.idx` files. The file name is in the format of `<output-prefix>_text_document.bin` (the suffix `_text_document` is added by default in the script). In the YAML file on this page, `train_dataset.dataloader.config.data_path` must be set to the **prefix containing the suffix**. For example, if `--output-prefix /path/megatron_data` is used, `/path/megatron_data_text_document.bin` will be generated. In this case, `/path/megatron_data_text_document` will be filled in the configuration. In addition, record the `eod` (eos) token ID corresponding to the tokenizer used during preprocessing, which will be used in the following `config.eod`.

## Step 1: Preparing the Configuration File

The dynamic graph uses the YAML configuration in dataclass style. The top-level sections correspond to the weight, training, parallelism, optimizer, learning rate, data, and model. This page provides a complete example configuration for two devices [`pynative_ds3.yaml`](https://atomgit.com/mindspore/docs/blob/r2.10.0/docs/mindformers/docs/source_en/example/quick_start/pynative_ds3.yaml), which can be directly downloaded and used. The content of each section is as follows (for details about the complete field description, see "Configuration File Description"):

```yaml
checkpoint:
  enable_save: False              # Do not save the weight for quick verification.
  save_path: "./output/ds3"

training:
  steps: 10                       # Number of training steps.
  local_batch_size: 1             # Batch size per device (per forward pass).
  global_batch_size: 2            # Global batch size. For details, see the following description.
  max_norm: 1.0                   # Gradient clipping (valid when the value is greater than 0).
  seed: 42

parallelism:
  data_parallel_shard: -1         # FSDP. The value -1 indicates automatic sharding based on the number of available devices.
  expert_parallel: 1
  tensor_parallel: 1
  context_parallel: 1
  pipeline_parallel: 1
  # SP is automatically enabled with TP. Currently, it cannot be disabled. (You do not need to configure this item. For details, see the document on distributed parallel training.)

optimizer:
  type: AdamW
  betas: [0.9, 0.95]
  eps: 1.e-8
  weight_decay: 0.01

lr_scheduler:
  type: ConstantWarmUpLR
  learning_rate: 1.e-5
  warmup_ratio: 0

train_dataset:
  dataloader:
    type: BlendedMegatronDatasetDataLoader
    datasets_type: "GPTDataset"
    sizes: [1000, 0, 0]           # Number of samples for [Training, Testing, Evaluation]. Currently, this parameter takes effect only for the training set.
    column_names: ["input_ids", "labels", "loss_mask", "position_ids"]
    shuffle: false
    config:
      seed: 1234
      seq_length: 4096            # The value must be the same as that of model.seq_length.
      split: "1, 0, 0"            # Split based on [Training, Testing, Evaluation]. This parameter is required if data_path is set. If this parameter is missing, an error will be reported.
      eod: 1                      # Token ID of the eod(eos) in the dataset.
      pad: -1                     # Token ID of the pad in the dataset.
      eod_mask_loss: False
      reset_position_ids: False
      create_attention_mask: False  # Four columns are output. If this parameter is set to True, add attention_mask to column_names.
      reset_attention_mask: False
      create_compressed_eod_mask: False
      eod_pad_length: 128
      data_path:                  # Sampling weight (relative value, automatically normalized) + bin prefix (containing _text_document).
        - '1'
        - "/path/megatron_data_text_document"
  drop_remainder: True
  num_parallel_workers: 8

model:
  model_type: deepseek_v3
  architectures: DeepseekV3ForCausalLM
  seq_length: 4096                # The value must be the same as that of train_dataset.dataloader.config.seq_length.
  # For details about other model structure hyperparameters (such as hidden_size, num_hidden_layers, and MoE), see the complete example configuration (tile format) in the preceding link.

```

> Precise meaning of `global_batch_size`: The framework infers the number of gradient accumulation steps based on `num_accumulation_steps = global_batch_size // (data_parallel x local_batch_size)`. When gradient accumulation is not enabled (that is, the product of the three values is exactly equal), `global_batch_size` is equal to `local_batch_size x Data parallelism degree`. Once `global_batch_size` is greater than this product, the excess multiple is the number of gradient accumulation steps. For the exact definition, see "Configuration File Description."

### Dataset Segments

When `BlendedMegatronDatasetDataLoader` is running, all of the following are `datasets_type`, `sizes`, and nested `config` blocks (including `seq_length`/`split`/`eod`/`pad`/`data_path`/`create_compressed_eod_mask`) **are required**.

| Field| Description|
|---|---|
| `datasets_type` | Dataset type. `"GPTDataset"` is used for pre-training.|
| `sizes` | Number of samples for `[Training, Testing, Evaluation]`. Currently, this parameter takes effect only for the training set.|
| `config.seq_length` | Length of the returned sequence. **This value must be the same as that of `model.seq_length`.**|
| `config.split` | Training/Testing/Evaluation sharding ratio (for example, `"1, 0, 0"`). This parameter is required if `data_path` is set.|
| `config.eod` / `config.pad` | Token ID of EOD (EOS)/pad, which is obtained from the tokenizer during preprocessing.|
| `config.data_path` | List. Every two elements (sampling weight and bin prefix) form a group. The weight is a relative value and is automatically normalized (the sum does not need to be 1). The bin prefix contains the `_text_document` suffix.|

> For details about the meaning of each field, multi-data source combination, and scenario-specific configurations such as compressed EOD mask, see "Dataset." This page provides only the minimum configuration.

### Model Section

The `model` section of DeepSeek-V3 contains dozens of structure hyperparameters (`hidden_size`, `num_hidden_layers`, MoE routing, etc.), making manual writing both tedious and error-prone. The complete example configuration [`pynative_ds3.yaml`](https://atomgit.com/mindspore/docs/blob/r2.10.0/docs/mindformers/docs/source_en/example/quick_start/pynative_ds3.yaml) provided on this page already includes all `model` section fields (dataclass style and structure hyperparameter tiling). You can directly download and use it. You only need to ensure that `seq_length` is consistent with the dataset.

> Note: The configuration under `configs/deepseek3/` is the static graph legacy structure (the structure hyperparameters are nested under `model.model_config`, `architectures` is a list, and `context.mode` is set to `0`). You cannot replicate the entire section to the dynamic graph configuration. Instead, you need to move the structure hyperparameters up one level and tile them under `model:`, and change `architectures` to a string.

## Step 2: Starting Training

Dynamic graph training uses `msrun` to start multiple devices. Run the following command in the **root directory of the MindFormers repository** (place the downloaded `pynative_ds3.yaml` file in this directory) to start training on two devices:

```bash
msrun --worker_num=2 --local_worker_num=2 --master_port=8118 \
      --join=True --log_dir=./msrun_log \
      run_mindformer.py --config pynative_ds3.yaml --mode 1
```

- `--worker_num`/`--local_worker_num`: Total number of devices/Number of devices on the local node.
- `--config`: YAML file in the previous step.
- `--mode 1`: uses dynamic graphs.

> To enter dynamic graph training, **`--mode 1` must be explicitly passed in the startup command**. (The entry reads only the command line `--mode` and does not read the YAML file.) The `context` section in the YAML file can be omitted. By default, the dynamic graph mode (`mode: 1`) and Ascend backend are used. If you need to adjust `max_device_memory`, explicitly configure it.

For details about how to start clusters of different scales (single-device and multi-node), see "Starting Tasks."

## Step 3: Viewing the Result

The training logs are output to the specified directory in `--log_dir`. Each worker has a subdirectory (for example, `./msrun_log/worker_0.log`). Open any worker log. If the following output is displayed, the training is running properly:

```text
{ step:[    1/   10], loss:  11.813965, per_step_time:  13570ms, load_balancing_loss:   1.093977, lr: 1.000000e-05, grad_norm:  13.831877, throughput:   1.36T }
{ step:[    2/   10], loss:  11.755612, per_step_time:    710ms, load_balancing_loss:   1.106543, lr: 1.000000e-05, grad_norm:  19.926786, throughput:  25.92T }
```

Key points for judgment:

- `step` continuously increases, and `loss` generally decreases (fluctuations may occur in the first few steps).
- `grad_norm` is a finite value rather than `NaN`.
- `per_step_time` tends to be stable after the first step (the first step includes initialization overhead and is usually significantly slower).
- The MoE model (such as DeepSeek-V3 in this example) will additionally print `load_balancing_loss`.

In addition:

- **Weight**: If `checkpoint.enable_save` is enabled, the weight is saved to `checkpoint.save_path` in the Safetensors format.
- **More metrics**: You can use `monitor.train_state` to collect per-parameter norms and local/device-level loss. (Note: Currently, dynamic graph monitoring metrics are output only through training logs, and the TensorBoard configuration does not take effect.) For details, see "Training Metric Monitoring and Profiling."

## Related Documents

- Overall architecture of dynamic graphs: [Overall Architecture](../introduction/overview.md)
- Currently supported models: [Model Support Library](../introduction/models.md)
- Overview of all capabilities: [Feature Overview](../feature/overview.md)
