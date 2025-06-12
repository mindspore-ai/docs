# Resumable Training After Breakpoint

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/feature/resume_training.md)

## Resumable Training

### Overview

MindSpore Transformers supports **step-level resumable training**, which allows the checkpoints of a model to be saved during training. If the training is interrupted, you can load a saved checkpoint to resume the training. This feature is crucial for processing large-scale training tasks, and can effectively reduce time and resource waste caused by unexpected interruptions. In addition, to resume a training where the dataset remains unchanged but the `global batch size` is changed, for example, when the cluster is changed or the configuration is modified, this tool supports automatic scaling of the number of resumable training steps and skipped data steps in the same proportion.

### Configuration and Usage

#### YAML Parameters

You can modify the configuration file to control resumable training. The main parameters are as follows. For details about other parameters, see the description of CheckpointMonitor.

| Parameter              | Description                                                                 |
|------------------|---------------------------------------------------------------------|
| load_checkpoint  | Weight path loaded during resumable training. The path can be a folder path (used to load distributed weights) or a specific weight file path. The default value is an empty string, indicating that no weight is loaded (required for resumable training). |
| resume_training  | Specifies whether to enable resumable training. You can set it to `True` or specify a weight file name. If the value is `True`, the system automatically resumes the training from the last interruption. The default value is `False`.   |
| load_ckpt_async | Determines whether to load model weights and compile in parallel (this configuration does not take effect when auto_trans_ckpt is set to true). The default value is False (serial execution). <br /> When it is `True`, the parallel capability of loading ckpt weights and building model is enabled to reduce the overall time  resume training. |

Based on the input parameters, there are four cases.

| load_checkpoint | resume_training | Description                                                                                                                                                                   | Recommended or Not|
|-----------------|-----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| Weight file path         | True            | Resumes a training based on the weights specified by load_checkpoint.                                                                                                                                               | √         |
| Weight file path         | Weight file name          | The file name specified by resume_training is invalid. A training is resumed based on the weights specified by load_checkpoint.                                                                                                                       | ×         |
| Weight folder path        | True            | **Scenario 1: Single-node system, multi-node system+shared directory, or ModelArts**<br>1. Resumes the training based on the weights recorded in meta.json files and supports fault recovery.<br>2. Resumes the training based on the latest weight of all ranks if the meta.json file of any rank is missing.<br>**Scenario 2: Multi-node+non-shared directory**<br>Resumes the training based on the latest weight of all ranks.| √         |
| Weight folder path        | Weight file name          | Resumes the training based on the weights specified by resume_training.                                                                                                                                               | √         |

In addition, you can modify the following parameters in the configuration file to use related functions.

| Parameter              | Description                                                                                                         |
|------------------|-------------------------------------------------------------------------------------------------------------|
| ignore_data_skip | Specifies whether to ignore the mechanism of skipping data during resumable training and read the dataset from the beginning instead. This parameter is used when the dataset is changed during resumable training. If this parameter is set to `True`, no data is skipped. The default value is `False`.                                    |
| data_skip_steps  | Number of steps skipped for the dataset. This parameter is used when the training is interrupted again after being resumed because the dataset or `global batch size` is changed. You need to manually set this parameter to configure the number of steps skipped for the new dataset. If the `global batch size` is changed, you need to divide and round down its value by the scaling coefficient and then specify the result as the value of this parameter.|

#### Fault Recovery Mechanism

If `resume_training` is set to `True`, the system automatically resumes training based on the weights recorded in `meta.json`. If the weight file of a rank is missing or damaged, the system rolls back to the latest available weight for recovery.

> In a distributed environment, resumable training requires that the weights of all nodes be in the same shared directory. You can use the `SHARED_PATHS` environment variable to set the shared path.

### Example of Distributed Training

The following example shows how to enable resumable training in single-device and multi-device environments. The example is based on the `llama2_7b` model.
For related configuration files, see [configs/llama2/pretrain_llama2_7b.yaml](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama2/pretrain_llama2_7b.yaml).

#### Complete Training

1. Modify `configs/llama2/pretrain_llama2_7b.yaml`.

   Configure the parallelism as required.

   ```yaml
   parallel_config:
     data_parallel: 1
     model_parallel: 2
     pipeline_stage: 2
     micro_batch_num: 2
   ```

   Configure the model weight saving as required.

   ```yaml
   callbacks:
     ...
     - type: CheckpointMonitor
       prefix: "llama2_7b"
       save_checkpoint_steps: 10
       keep_checkpoint_max: 3
       integrated_save: False
       async_save: False
     ...
   ```

2. Prepare a dataset. The following uses [wikitext2](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama2.md#%E6%95%B0%E6%8D%AE%E5%8F%8A%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87) as an example to describe how to start four-device distributed training.

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
       --config configs/llama2/pretrain_llama2_7b.yaml \
       --train_dataset /path/to/wikitext2-llama2.mindrecord \
       --run_mode train \
       --use_parallel True" 4
   ```

   After the fourth saving is complete, end the process. The structure of the `rank_0` folder under `checkpoint` is as follows:

   ```text
   checkpoint/rank_0
     ├── llama2_7b_rank_0-10_2.ckpt
     ├── llama2_7b_rank_0-15_2.ckpt
     ├── llama2_7b_rank_0-20_2.ckpt
     └── meta.json
   ```

#### Resumable Training

1. Modify the configuration and specify the resumable training weight file.

   ```yaml
   load_checkpoint: './output/checkpoint'
   resume_training: True
   ```

2. Resume training.

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
       --config configs/llama2/pretrain_llama2_7b.yaml \
       --train_dataset /path/to/wikitext2-llama2.mindrecord \
       --run_mode train \
       --use_parallel True" 4
   ```

   If the initial number of steps is `42`, the training is resumed successfully. The saved weight file contains the information about step `40`. The default value of `sink_size` is `2`, indicating that the information is printed every two steps. Therefore, the initial number of steps is `42`.

#### Resumable Training with the Dataset Changed

There are three main scenarios where the dataset is changed in resumable training. You need to modify the configuration file in each scenario. The following describes each case one by one, and describes in detail which step of the basic resumable training process needs to be modified, and how to modify a specific configuration to achieve an expected effect.

**Scenario 1: Training resumed with a new dataset (but not skipping trained steps)**

In this scenario, when the new dataset is used, the model training starts from scratch without skipping any data or steps. In this case, you need to set the configuration file **to ignore the previous data progress** so that the model can be trained from scratch based on the new dataset.

- **Configuration modification**: You need to set `ignore_data_skip` based on the first step of the basic resumable training process. Set `ignore_data_skip` to `True`, indicating that no data is skipped.

   ```yaml
   load_checkpoint: './output/checkpoint'
   resume_training: True
   ignore_data_skip: True
   ```

- **Expected result**: The model is trained from scratch based on the new dataset without skipping any steps.

**Scenario 2: Training resumed with a new dataset, skipping trained steps**

In this case, the model has been partially trained based on the new dataset (for example, `2` steps have been performed before the training is interrupted), and the training is expected to continue from the last interruption. In this case, you must manually specify the number of steps to be skipped.

- **Configuration modification**: You need to set `ignore_data_skip` and `data_skip_steps` based on the first step of the basic resumable training process. Set `ignore_data_skip` to `False` and use `data_skip_steps` to specify the number of trained steps to skip (for example, `2`).

   ```yaml
   load_checkpoint: './output/checkpoint'
   resume_training: True
   ignore_data_skip: False
   data_skip_steps: 2
   ```

- **Expected result**: The model skips the first `2` steps and continues the training from step `3` based on the new dataset.

**Scenario 3: Training resumed with a new dataset and `global batch size` changed**

If `global batch size` is changed (for example, doubled) when a training is resumed based on a new dataset, you need to scale the number of steps that have been performed when manually specifying the number of steps to be skipped. Specifically, the number of skipped steps needs to be divided and rounded down based on the scaling coefficient. For example, if the value of `global batch size` is changed to `2` times of the original value, the number of steps that need to be skipped is halved.

- **Configuration modification**: Adjust `data_skip_steps` based on Scenario 2. Set `data_skip_steps` to the number of steps after scaling. For example, if `global batch size` is changed to `2` times of the original value, the number of steps to be skipped is changed to `1` (rounded down).

   ```yaml
   load_checkpoint: './output/checkpoint'
   resume_training: True
   ignore_data_skip: False
   data_skip_steps: 1
   ```

- **Expected result**: The model adjusts the number of skipped steps based on the new setting of `global batch size` and continues the training from the specified position.

#### Fault Recovery Example

If some weight files are missing, the system automatically restores the files based on the latest available weight.

1. Delete the `llama2_7b_rank_0-20_2.ckpt` file from the `rank_3` directory. The folder structure after the deletion is as follows:

   ```text
   checkpoint/rank_3
     ├── llama2_7b_rank_0-10_2.ckpt
     ├── llama2_7b_rank_0-15_2.ckpt
     └── meta.json
   ```

2. Modify the configuration to enable fault recovery.

   ```yaml
   load_checkpoint: './output/checkpoint'
   resume_training: True
   ```

3. Start distributed training.

   ```shell
   bash scripts/msrun_launcher.sh "run_mindformer.py \
       --config configs/llama2/pretrain_llama2_7b.yaml \
       --train_dataset /path/to/wikitext2-llama2.mindrecord \
       --run_mode train \
       --use_parallel True" 4
   ```

   If the initial number of steps is `32`, the training is resumed successfully. Because the weight of the information in step `40` under `rank_3` is deleted, the weight saved last time, that is, the weight of the information in step `30`, is automatically used. The default value of `sink_size` is `2`, indicating that information is printed every two steps. Therefore, the initial number of steps is `32`.

### Precautions

- **Data offloading**: You must enable data offloading and configure `sink_mode=True` for distributed resumable training.
- **Weight file check**: Ensure that the weights loaded for resumable training are the ones saved when the training is interrupted instead of in the entire training process. Otherwise, an error is reported.
