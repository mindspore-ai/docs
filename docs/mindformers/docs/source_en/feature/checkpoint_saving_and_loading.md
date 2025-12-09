# Checkpoint Saving and Loading

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/checkpoint_saving_and_loading.md)

## Overview

MindSpore Transformers supports saving intermediate checkpoints during training. Checkpoints include components such as **model weights**, **optimizer weights**, **training context information**, and **distributed strategy meta-information**. Their core functions are to **resume training after interruption**, **prevent progress loss due to training failures**, and support **subsequent fine-tuning**, **inference**, or **model iteration**.

MindSpore Transformers has launched **Checkpoint 2.0**, which achieves comprehensive improvements in usability and loading efficiency by reconstructing the checkpoint saving strategy and loading process.

Compared with Checkpoint 1.0, the core updates are as follows:

- **New checkpoint saving [directory structure](#directory-structure)**: The checkpoint directory contains independent files for **model weights**, **optimizer weights**, **training context information**, **distributed strategy meta-information**, etc.;
- **Added online Reshard loading mechanism**: If the distributed strategy meta-information of the checkpoint to be loaded is inconsistent with the current task, Reshard conversion will be **automatically performed on the weight parameters** during loading to generate parameters adapted to the current distributed strategy;
- **Simplified loading configuration**: Relying on the online Reshard mechanism, users **do not need to manually configure parameters such as `auto_trans_ckpt` and `src_strategy_path_or_dir`** to trigger weight strategy conversion, which significantly improves usability.

MindSpore Transformers currently uses Checkpoint 1.0 by default. Users need to add the following parameters to the YAML configuration file to enable the saving and loading functions of Checkpoint 2.0.

```yaml
use_legacy_format: False
```

> This document is only for users to experience Checkpoint 2.0. If using Checkpoint 1.0, please refer to the [Safetensors Document](https://www.mindspore.cn/mindformers/docs/en/master/feature/safetensors.html) or [Ckpt Document](https://www.mindspore.cn/mindformers/docs/en/master/feature/ckpt.html).

## Checkpoint Saving

### Directory Structure

The training checkpoints of MindSpore Transformers are stored in the `output/checkpoint` directory by default, and each checkpoint is independently saved as a subfolder named after `iteration`. Taking the checkpoint generated in the first step of an 8-card task as an example, its saving format is as follows:

```text
output
    ├── checkpoint
        ├── iteration_00000001
            ├── metadata.json
            ├── common.json
            ├── {prefix}-model-0000000-0000008.safetensors
            ...
            ├── {prefix}-model-0000007-0000008.safetensors
            ├── {prefix}-opt-0000000-0000008.safetensors
            ...
            └── {prefix}-opt-0000007-0000008.safetensors
        ...
        └── latest_checkpointed_iteration.txt
```

Description of weight-related files

| File                                       | Description                                                  |
| ------------------------------------------ | ------------------------------------------------------------ |
| metadata.json                              | Records the distributed strategy meta-information and storage information of each parameter, providing necessary metadata support for automatically performing Reshard conversion when loading weights later, ensuring that the conversion is accurately adapted to the current task. |
| common.json                                | Records the training information of the current iteration, providing data support for resuming training from a breakpoint. |
| {prefix}-model-0000000-0000008.safetensors | Model weight storage file. Naming rule description: `prefix` is a custom file name prefix, `model` identifies the file type as model weights, `0000000` is the file sequence number, and `0000008` represents the total number of files. |
| {prefix}-opt-0000000-0000008.safetensors   | Optimizer weight storage file. Naming rule description: `prefix` is a custom file name prefix, `opt` identifies the file type as optimizer weights, `0000000` is the file sequence number, and `0000008` represents the total number of files. |
| latest_checkpointed_iteration.txt          | Records the iteration step corresponding to the last successfully saved checkpoint in the `output/checkpoint` directory. |

### Configuration Instructions

Users can control the weight saving behavior by modifying the relevant fields under `CheckpointMonitor` in the YAML configuration file. The specific parameter descriptions are as follows:

| Parameter Name        | Description                                                  | Value Description                                            |
| --------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| prefix                | Custom prefix for weight file names. It is recommended to fill in the model name to distinguish checkpoints of different models. | (str, optional) - Default value: `"CKP"`.                    |
| directory             | The path where checkpoints are saved. If not configured, they are stored in `./output/checkpoint` by default. | (str, optional) - Default value: `None`.                     |
| save_checkpoint_steps | Set the training interval steps for saving checkpoints (i.e., save a checkpoint every specified number of training steps). | (int, optional) - Default value: `1`. If not set, model weights will not be saved. |
| keep_checkpoint_max   | Set the maximum number of checkpoints to keep. When the limit is reached, the oldest checkpoint will be automatically deleted when a new checkpoint is saved. | (int, optional) - Default value: `5`.                        |
| async_save            | Switch for the asynchronous checkpoint saving function (controls whether to enable the asynchronous saving mechanism). | (bool, optional) - When `True`, an asynchronous thread will be used to save checkpoints. Default value: `False`. |
| checkpoint_format     | The saving format of checkpoint weights. Checkpoint 2.0 only supports `'safetensors'`; if `use_legacy_format: False` is configured, this field will be automatically converted to `'safetensors'`. | (str, optional) - Default value: `'safetensors'`.            |
| remove_redundancy     | Switch for the checkpoint redundancy removal function (controls whether to enable the redundancy removal saving mechanism). | (bool, optional) - Default value: `False`.                   |
| save_optimizer        | Switch for the optimizer weight saving function (controls whether to save optimizer weight information). | (bool, optional) - Default value: `True`.                    |

Configuration example is as follows:

```yaml
callbacks:
  ...
  - type: CheckpointMonitor
    prefix: "qwen3"
    save_checkpoint_steps: 1000
    keep_checkpoint_max: 5
    async_save: False
    checkpoint_format: "safetensors"
    save_optimizer: True
  ...
```

> The above configuration specifies that the training task uses "qwen3" as the prefix for safetensors file names, adopts the synchronous saving mode, saves checkpoints containing model weights and optimizer weights every 1000 steps, and retains at most the latest 5 checkpoints throughout the training process.

If you want to learn more about CheckpointMonitor, you can refer to the [CheckpointMonitor API Document](https://www.mindspore.cn/mindformers/docs/en/master/core/mindformers.core.CheckpointMonitor.html).

## Checkpoint Loading

MindSpore Transformers provides flexible checkpoint loading capabilities, covering all scenarios of single-card and multi-card, with the following core features:

1. Adaptability upgrade for Checkpoint 2.0: Relying on the online Reshard mechanism, weights can be automatically adapted to any distributed strategy task during loading without manual adjustment, reducing the cost of multi-scenario deployment;
2. Cross-platform weight compatibility: Through a dedicated conversion interface, it supports loading weight files released by the HuggingFace community. Currently, it has achieved compatible adaptation for the Qwen3 model training scenario, facilitating users to reuse community resources.

### Configuration Instructions

Users can control the weight loading behavior by modifying the relevant fields in the YAML configuration file.

| Parameter Name       | Description                                                  | Value Description                         |
| -------------------- | ------------------------------------------------------------ | ----------------------------------------- |
| load_checkpoint      | The path to the checkpoint folder, supporting **filling in the `output/checkpoint` folder path** or **the specific `iteration` subfolder path**; if the former is filled in, the checkpoint in the corresponding `iteration` subfolder will be loaded according to the step recorded in `latest_checkpointed_iteration.txt`. | (str, optional) - Default value: `""`     |
| pretrained_model_dir | Specify the folder path of HuggingFace community weights; if `load_checkpoint` is also configured, this field will be automatically invalidated. | (str, optional) - Default value: `""`     |
| balanced_load        | Switch for the weight balanced loading function, **only supported in distributed tasks**; when set to `True`, each rank loads weights according to the parameter balanced allocation strategy, and then obtains the final weights through parameter broadcasting. | (bool, optional) - Default value: `False` |
| use_legacy_format    | Switch for enabling Checkpoint 1.0, which needs to be set to `False` (i.e., using Checkpoint 2.0 by default). | (bool, optional) - Default value: `True`  |
| load_ckpt_format     | Specify the format of the loaded weights, which needs to be set to `'safetensors'` (to adapt to Checkpoint 2.0). | (bool, optional) - Default value: `ckpt`  |

When `load_checkpoint` is configured as the path of the `output/checkpoint` folder, users can modify the step recorded in `latest_checkpointed_iteration.txt` to load the weights of the specified `iteration`.

## Constraint Description

- In multi-machine scenarios, all files need to be stored in the **same shared directory**, and users need to configure the **shared path to the environment variable `SHARED_PATHS`**. It is recommended to configure it as the uppermost shared directory path first. Example: If the shared directory is `/data01` (the project directory is under it), you can execute `export SHARED_PATHS=/data01`.