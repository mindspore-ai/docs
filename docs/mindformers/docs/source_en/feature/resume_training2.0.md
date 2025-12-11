# Resume Training2.0

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/resume_training2.0.md)

## Overview

MindSpore Transformers has complete resume training capabilities. The core functions and applicable scenarios are as follows:

1. **Core Functions**: Supports loading saved checkpoints to quickly resume training progress without starting from scratch;
2. **Multi-scenario Adaptation**: Covers four mainstream resume training scenarios
   - **Interruption Resume Training**: After an abnormal interruption of a normal training task (such as equipment failure, network fluctuation), resume the training process based on the saved checkpoint;
   - **Scaling Resume Training**: Adjust the number of cards (expansion / reduction) during training and continue training based on the saved checkpoint;
   - **Incremental Resume Training**: On the basis of existing training results, supplement the training dataset and continue training based on the saved checkpoint;
   - **Automatic Recovery Resume Training**: Supports the platform to automatically start resume training without manual intervention;

For large-scale training tasks (long training cycles and large resource investment), it can avoid progress loss caused by unexpected interruptions and significantly reduce time and computing resource waste.

> This document only applies to scenarios where [Checkpoint 2.0](https://www.mindspore.cn/mindformers/docs/en/master/feature/checkpoint_saving_and_loading.html) are used for resume training; if users use Checkpoint 1.0, please refer to the old version [resume training document](https://www.mindspore.cn/mindformers/docs/en/master/feature/resume_training.html).

## Checkpoint Introduction

The training checkpoints of MindSpore Transformers are stored in the `output/checkpoint` directory by default, and each checkpoint is independently saved as a subfolder named after `iteration`. Taking the checkpoint generated in the first step of an 8-card task as an example, its saving format is as follows:

```text
output
    ├── checkpoint
        ├── iteration_0000001
            ├── metadata.json
            ├── common.json
            ├── {prefix}-model-0000000-0000008.safetensor
            ...
            ├── {prefix}-model-0000007-0000008.safetensor
            ├── {prefix}-opt-0000000-0000008.safetensor
            ...
            └── {prefix}-opt-0000007-0000008.safetensor
        ...
        └── latest_checkpointed_iteration.txt
```

You can refer to [Checkpoint Saving and Loading](https://www.mindspore.cn/mindformers/docs/en/master/feature/checkpoint_saving_and_loading.html) for more information about checkpoints.

## Configuration Description

| Parameter Name  | Description                                                  | Value Description                         |
| --------------- | ------------------------------------------------------------ | ----------------------------------------- |
| load_checkpoint | The path to the checkpoint folder. It can **be filled with the path of the `output/checkpoint` folder or the path of the `iteration` subfolder**.<br />If it is the path of the `checkpoint` folder, the checkpoint in the corresponding `iteration` subfolder will be loaded according to the number of iterations recorded in `latest_checkpointed_iteration.txt`. | (str, optional) - Default value: `""`     |
| resume_training | The switch for the resume training function. When set to `True`, training will restore from the number of iterations corresponding to the checkpoint to be loaded. | (bool, optional) - Default value: `False` |

## Scenario Introduction

### Interruption Resume Training

**Overview**: After an abnormal interruption of a normal training task, resume the training process based on the saved checkpoint without changing the distributed strategy.

MindSpore Transformers provides two ways to start resuming training:

- Resume training based on the number of iterations recorded in `latest_checkpointed_iteration.txt`

  ```yaml
  load_checkpoint: /path/to/checkpoint
  resume_training: True
  ```

- Resume training based on the specified number of iterations

  ```yaml
  load_checkpoint: /path/to/checkpoint/iteration_{x}
  resume_training: True
  ```

  > x represents the training iteration step corresponding to the checkpoint. For example, "0000001" indicates the checkpoint corresponding to the 1st training step.

### Scaling Resume Training

**Overview**: When it is necessary to **expand/reduce the cluster scale** or **modify the distributed strategy** to continue the training task, the configuration is the same as [Interruption Resume Training](#interruption-resume-training). Relying on the online Reshard mechanism, MindSpore Transformers can ensure that the checkpoint weights automatically adapt to any distributed strategy, ensuring smooth resume training.

- Resume training based on the number of iterations recorded in `latest_checkpointed_iteration.txt`

  ```yaml
  load_checkpoint: /path/to/checkpoint
  resume_training: True
  ```

- Resume training based on the specified number of iterations

  ```yaml
  load_checkpoint: /path/to/checkpoint/iteration_{x}
  resume_training: True
  ```

  > x represents the training iteration step corresponding to the checkpoint. For example, "0000001" indicates the checkpoint corresponding to the 1st training step.

### Incremental Resume Training

**Overview**: The training dataset needs to be **produced and trained simultaneously**. After the current dataset is trained, add the newly produced dataset to continue training until all datasets are trained. This scenario requires users to preset the total steps of the learning rate curve in advance according to the total amount of data for training.

Assume that a total of 10T tokens of data are trained, each produced dataset contains only 1T tokens of data, and the entire training process is completed in 10 epochs, which takes a total of 100000 steps.

- Step 1: Preset the total training steps to fix the learning rate curve of the entire training process

  ```yaml
  lr_schedule:
    total_steps: 100000
  ```

- Step 2: Set a sufficiently large epoch value to ensure that all datasets can be trained

  ```yaml
  runner_config:
    epochs: 15
  ```

  > The learning rate curve of the entire training process has been fixed, and the epoch value setting will not affect the learning rate. A larger value can be set to ensure that 10 datasets can be trained.

- Step 3: After training one epoch of the dataset, you can replace the dataset to resume training. The following is resume training based on the number of iterations recorded in `latest_checkpointed_iteration.txt`. For other resume training methods, please refer to [Interruption Resume Training](#interruption-resume-training) or [Scaling Resume Training](#scaling-resume-training).

  ```yaml
  load_checkpoint: /path/to/checkpoint
  resume_training: True
  ```

  > When replacing the dataset for resume training, due to the different number of samples in each dataset, the displayed epoch and single-batch step may change, but the total number of training steps remains unchanged, which is a normal phenomenon.

### Automatic Recovery Resume Training

**Overview**: To support the platform to automatically start resume training without manual intervention, `load_checkpoint` can be configured as the checkpoint saving directory path: when training for the first time, the directory is empty, and the model initializes parameters randomly; during resume training, it will recover training based on the last saved complete checkpoint in the directory.

```yaml
load_checkpoint: /path/to/output/checkpoint
resume_training: True
```

## Constraint Description

- In multi-machine scenarios, all checkpoint files need to be stored in the same shared directory for resume training. Users need to configure the shared path to the environment variable `SHARED_PATHS`; it is recommended to configure the top-level shared directory first. Example: When the shared directory is `/data01`, execute `export SHARED_PATHS=/data01`.
