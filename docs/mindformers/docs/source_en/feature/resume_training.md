# Resumable Training After Breakpoint

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/resume_training.md)

## Overview

MindSpore Transformers supports **step-level resume training** functionality, enabling the loading of saved checkpoints to resume previous training states. This feature is particularly important for handling large-scale training tasks, as it effectively reduces time and resource waste caused by unexpected interruptions.

MindSpore Transformers supports saving and loading weights in both **ckpt** and **safetensors** formats. It supports various resume training scenarios such as **interrupted training resumption**, **strategy conversion resumption**, **incremental training resumption**, and **automatic recovery resumption**. It also supports different weight loading methods including **loading the last fully saved weights**, **loading weights from a specified step**, and **loading MindSpore merged weights** for resumption.

In a distributed environment, resume training requires that weights from all nodes be stored in the **same shared directory**. Users can set the shared path via the environment variable `SHARED_PATHS`.

## Introduction to Weight and Strategy Files

MindSpore Transformers saves weight and strategy files, which are by default stored in the `output/checkpoint` and `output/strategy` folders. Users can modify the `output_dir` parameter in the YAML configuration to change the path of the `output` folder.

Weight files mainly store **network parameters**, **optimizer parameters**, and **resume training information**. Weight files are saved separately in rank-specific folders, and each rank folder maintains a `meta.json` file to record the last fully saved weight information for that rank. Taking a single-machine 8-card setup as an example, the weight saving format is as follows:

```text
output/checkpoint
    ├── rank_0
      ├── meta.json
      └── {prefix}-{epoch}_{step}.safetensors
    ├── rank_1
      ├── meta.json
      └── {prefix}-{epoch}_{step}.safetensors
    ...
    ├── rank_7
      ├── meta.json
      └── {prefix}-{epoch}_{step}.safetensors
```

> The prefix of the weight name contains rank_id information, e.g., `llama3_1_8b_rank_0`. If a weight with the same prefix already exists when saving, an incremental suffix will be automatically added to the prefix to prevent overwriting old weights. For example, if "llama3_1_8b_rank_0" already exists, the prefix will be updated to "llama3_1_8b_rank_0_1", and if "llama3_1_8b_rank_0_1" also exists, it will be updated to "llama3_1_8b_rank_0_2".

Strategy files are only saved in distributed training tasks and are used for **weight strategy conversion**. Strategy files are saved in ckpt format with the rank_id as the suffix, mainly recording the network and optimizer sharding information for the current rank. Taking a single-machine 8-card setup as an example, the strategy file saving format is as follows:

```text
output/strategy
    ├── ckpt_strategy_rank_0.ckpt
    ├── ckpt_strategy_rank_1.ckpt
    ...
    └── ckpt_strategy_rank_7.ckpt
```

> Strategy files will overwrite old files when saved. To prevent overwriting or mixing strategy files from different tasks, please promptly save strategy files to a custom folder.

For more information about weights, refer to [Ckpt Weights](https://www.mindspore.cn/mindformers/docs/en/master/feature/ckpt.html) and [Safetensors Weights](https://www.mindspore.cn/mindformers/docs/en/master/feature/safetensors.html).

## YAML Parameter Configuration Description

| Parameter                | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| load_checkpoint          | Path to the weight file or folder, **required for resuming training**, default is an empty string.<br />If the configured path is an empty directory, it will fall back to using randomly initialized weights for pre-training.<br />For single-card weights, configure the path to the weight file, ensuring the parent directory does not start with "rank_". |
| src_strategy_path_or_dir | Path to the strategy file or folder, required when **`auto_trans_ckpt=True` and load_checkpoint is a distributed weight**, default is an empty string.<br />If the weights configured in load_checkpoint do not have pipeline parallel sharding, configure any strategy file path; otherwise, configure the strategy folder path. |
| auto_trans_ckpt          | Switch for automatic weight conversion, needs to be enabled when the **weights configured in load_checkpoint do not match the distributed strategy of the current task**, default is False. |
| transform_process_num    | Number of processes used for automatic weight conversion, **only applicable to automatic conversion of ckpt format weights**, which can accelerate weight conversion. Default is `None` (disabled).<br />The set value must be divisible by the total number of cluster cards. A larger value increases host memory usage; reduce the number of processes if host memory is insufficient. |
| resume_training          | Switch for resuming training, can be set to `True` or the weight file name in any rank sub-folder. Default is `False`.<br />When set to `True`, it **loads the last fully saved weights** for resumption.<br />When set to a weight file name, it **loads the weights from the specified step** for resumption. |
| load_ckpt_format         | Format of the weights configured in load_checkpoint, can be set to `safetensors` or `ckpt`, default is `ckpt`. |
| remove_redundancy        | Switch for loading without redundancy, needs to be enabled when the weights configured in load_checkpoint are **safetensors format weights saved without redundancy**, default is False. |
| load_ckpt_async          | Whether to execute weight loading in parallel with model compilation. This configuration **only applies to asynchronous loading scenarios with ckpt format weights and unchanged distributed strategy**. Default is `False`. |

## Introduction to Resume Training Scenarios

### Interrupted Training Resumption

**Overview**: Resume training based on saved weights after an unexpected interruption of a normal training task, without changing the distributed strategy.

- Resume training from the last fully saved weights

  ```yaml
  load_checkpoint: /path/to/checkpoint
  resume_training: True
  ```

  The system will automatically search for and load the last fully saved weights based on the weight records in each rank's `meta.json` for resumption.

  > If there is no meta.json in all rank sub-folders of the weight folder, it will fall back to resuming from the weights with the latest timestamp for each rank.

- Resume training from weights of a specified step

  ```yaml
  load_checkpoint: /path/to/checkpoint
  # For ckpt weights, fill in {prefix}-{epoch}_{step}.ckpt
  resume_training: {prefix}-{epoch}_{step}.safetensors
  ```

  Users must ensure the integrity of the specified weights. Each rank will automatically replace the rank information in the "prefix" to update the weight name to be loaded. For example, if the specified weight name is `llama3_1_8b_rank_0-200_1.safetensors`, when loading rank_1, the weight name will be replaced with `llama3_1_8b_rank_1-200_1.safetensors`. An error will occur if the weight is missing for a certain rank.

### Strategy Conversion Resumption

**Overview**: Continue training after modifying the **distributed strategy** or **expanding/shrinking the cluster scale**, requiring **enabling automatic weight conversion**.

#### Safetensors Weights

Enabling automatic weight conversion will automatically merge safetensors weights into [full weights](https://www.mindspore.cn/mindformers/docs/en/master/feature/safetensors.html#full-weights) for distributed loading. The merged safetensors weights will be saved to the `output/unified_checkpoint` folder. If the weights have been offline merged into [full weights](https://www.mindspore.cn/mindformers/docs/en/master/feature/safetensors.html#full-weights), they will be directly loaded in a distributed manner. For offline merging steps, refer to the [Safetensors Weights - Weight Slicing and Merging](https://www.mindspore.cn/mindformers/docs/en/master/feature/safetensors.html) section.

- Resume training from the last fully saved weights

  ```yaml
  load_checkpoint: /path/to/checkpoint
  src_strategy_path_or_dir: /path/to/strategy
  resume_training: True
  auto_trans_ckpt: True
  ```

- Resume training from weights of a specified step

  ```yaml
  load_checkpoint: /path/to/checkpoint
  src_strategy_path_or_dir: /path/to/strategy
  resume_training: {prefix}-{epoch}_{step}.safetensors
  auto_trans_ckpt: True
  ```

- Resume training from merged weights

  ```yaml
  load_checkpoint: /path/to/unified_checkpoint
  resume_training: True
  auto_trans_ckpt: True
  ```

#### Ckpt Weights

Enabling automatic weight conversion will automatically convert weights to the distributed strategy of the current task before loading. The converted ckpt weights will be saved to the `output/transformed_checkpoint` folder, which can be directly loaded for subsequent use without enabling weight automatic conversion.

If there are multiple step weight files in the rank sub-folder of the weights, it is necessary to offline filter the weights to ensure that **each rank sub-folder contains only a single ckpt file to be loaded**.

```yaml
load_checkpoint: /path/to/checkpoint
src_strategy_path_or_dir: /path/to/strategy
resume_training: True
auto_trans_ckpt: True
transform_process_num: 8
```

### Incremental Training Resumption

**Overview**: The training dataset needs to be **produced and trained incrementally**. After training on the current dataset, new produced datasets are added for continued training until all datasets are processed. This scenario requires users to preset the total steps of the learning rate curve in advance based on the total amount of training data.

Assume a total of 10T tokens of data will be trained, with each produced dataset containing 1T tokens. The entire training process is completed in 10 epochs, requiring a total of 100,000 steps.

- Step 1: Preset the total training steps to fix the learning rate curve for the entire training process

  ```yaml
  lr_schedule:
    total_steps: 100000
  ```

- Step 2: Set a sufficiently large epoch value to ensure all datasets can be trained

  ```yaml
  runner_config:
    epochs: 15
  ```

  > The learning rate curve for the entire training process is fixed, and the epoch value setting will not affect the learning rate. You can set a larger value to ensure that all 10 datasets are fully trained.

- Step 3: After training 1 epoch of the dataset, replace the dataset and resume training. The following example resumes from the last fully saved weights; for other resumption methods, refer to [Interrupted Training Resumption](#interrupted-training-resumption) or [Strategy Conversion Resumption](#strategy-conversion-resumption).

  ```yaml
  load_checkpoint: /path/to/checkpoint
  resume_training: True
  ```

  > Due to inconsistent sample counts across datasets, the displayed epoch and step may change when resuming with a new dataset. However, the total number of training steps remains unchanged, which is a normal phenomenon.

### Automatic Recovery Resumption

**Overview**: To facilitate automatic resumption of training by the platform without manual intervention, configure load_checkpoint to the save path of weight checkpoints. During the first training run, this directory is empty, and training will start normally with randomly initialized weights. For resumption, training will resume from the last fully saved weights in this directory.

```yaml
load_checkpoint: /path/to/output/checkpoint
resume_training: True
```

## Notes and Recommendations

- Distributed resume training must enable **data sinking mode** by configuring `sink_mode=True`.
- It is recommended to set the `SHARED_PATHS` environment variable to the path of the top-level shared directory. For example, if `/data01` is the shared directory and the project directory is under it, configure `export SHARED_PATHS=/data01`.
- It is recommended to save weights and strategy files of training tasks with different distributed strategies in separate folders.
