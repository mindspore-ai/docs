# Start Tasks

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_en/function/start_tasks.md)

## Overview

MindSpore Transformers provides a one-click startup script `run_mindformer.py` and a distributed task launch script `msrun_launcher.sh`.

- The `run_mindformer.py` script is used to start tasks on a **single device**, providing one-click capabilities for pre-training, fine-tuning, and inference tasks.
- The `msrun_launcher.sh` script is used to start distributed tasks on **multi-device within a single node** or **multi-device with multi-node**, launching tasks on each device through the [msrun](https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html) tool.

## Run_mindformer One-click Start Script

In the root directory of the MindSpore Transformers code, execute the `run_mindformer.py` script using Python to start the task. The supported parameters of the script are as follows. **When an optional parameter is not set or is set to ``None``, the configuration with the same name in the YAML configuration file will be taken**.

### Basic Parameters

|     Parameters      | Parameter Descriptions                                                                                                                                                                        | Value Description                                                                                   | Applicable Scenarios        |
|:-------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|-----------------------------|
|     `--config`      | YAML config files.                                                                                                                                                                            | str, required                                                                                       | pre-train/finetune/predict  |
|      `--mode`       | Set the backend execution mode.                                                                                                                                                               | int, optional, `0` is GRAPH_MODE and `1` is PYNATIVE_MODE. Currently, only GRAPH_MODE is supported. | pre-train/finetune/predict  |
|    `--device_id`    | Set the execution device ID. The value must be within the range of available devices.                                                                                                         | int, optional                                                                                       | pre-train/finetune/predict  |
|  `--device_target`  | Set the backend execution device. MindSpore Transformers is only supported on `Ascend` devices.                                                                                               | str, optional                                                                                       | pre-train/finetune/predict  |
|    `--run_mode`     | Set the running mode of the model: `train`, `finetune` or `predict`.                                                                                                                          | str, optional                                                                                       | pre-train/finetune/predict  |
| `--load_checkpoint` | File or folder paths for loading weights. For detailed usage, please refer to [Weight Conversion Function](https://www.mindspore.cn/mindformers/docs/en/dev/function/weight_conversion.html)  | str, optional                                                                                       | pre-train/finetune/predict  |
|  `--use_parallel`   | Whether use parallel mode.                                                                                                                                                                    | bool, optional                                                                                      | pre-train/finetune/predict  |
|   `--output_dir`    | Set the path where log, checkpoint, strategy, etc. files are saved.                                                                                                                           | str, optional                                                                                       | pre-train/finetune/predict  |
|  `--register_path`  | The absolute path of the directory where the external code is located. For example, the model directory under the research directory.                                                         | str, optional                                                                                       | pre-train/finetune/predict  |
|      `--seed`       | Set the global seed. For details, refer to [mindspore.set_seed](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.set_seed.html).                                        | int, optional                                                                                       | pre-train/finetune/predict  |

### Weight Slicing

|          Parameters          | Parameter Descriptions                                                                                                                                              | Value Description                                                                          | Applicable Scenarios        |
|:----------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|-----------------------------|
| `--src_strategy_path_or_dir` | The strategy of load_checkpoint.                                                                                                                                    | str, optional                                                                              | pre-train/finetune/predict  |
|     `--auto_trans_ckpt`      | Enable online weight automatic conversion. Refer to [Weight Conversion Function](https://www.mindspore.cn/mindformers/docs/en/dev/function/weight_conversion.html). | bool, optional                                                                             | pre-train/finetune/predict  |
|  `--transform_process_num`   | The number of processes responsible for checkpoint transform.                                                                                                       | int, optional                                                                              | pre-train/finetune/predict  |
|    `--only_save_strategy`    | Whether to only save the strategy files.                                                                                                                            | bool, optional, when it is `true`, the task exits directly after saving the strategy file. | pre-train/finetune/predict  |

### Training

|            Parameters            | Parameter Descriptions                                                                                                                                                                                      | Value Description | Applicable Scenarios |
|:--------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|----------------------|
|      `--train_dataset_dir`       | Dataset directory of data loader to pre-train/finetune.                                                                                                                                                     | str, optional     | pre-train/finetune   |
|       `--resume_training`        | Enable resumable training after breakpoint. For details, refer to [Resumable Training After Breakpoint](https://www.mindspore.cn/mindformers/docs/en/dev/function/resume_training.html#resumable-training). | bool, optional    | pre-train/finetune   |
|            `--epochs`            | Train epochs.                                                                                                                                                                                               | int, optional     | pre-train/finetune   |
|          `--batch_size`          | The sample size of the batch data.                                                                                                                                                                          | int, optional     | pre-train/finetune   |
| `--gradient_accumulation_steps`  | The number of gradient accumulation steps.                                                                                                                                                                  | int, optional     | pre-train/finetune   |
|         `--num_samples`          | Number of datasets samples used.                                                                                                                                                                            | int, optional     | pre-train/finetune   |

### Inference

|       Parameters       | Parameter Descriptions                                                 | Value Description                                                                                                                                             | Applicable Scenarios |
|:----------------------:|:-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
|    `--predict_data`    | Input data for predict.                                                | str, optional, It can be the input for predict (single-batch predict) or the file path of a txt file containing multiple lines of text (multi-batch predict). | predict              |
| `--predict_batch_size` | Batch size for predict data, set to perform batch predict.             | int, optional                                                                                                                                                 | predict              |
|     `--do_sample`      | Whether to use random sampling when selecting tokens when predicting.  | int, optional, ``True`` means using sampling encoding, ``False`` means using greedy decoding.                                                                 | predict              |

## Distributed Task Pull-up Script

The distributed task pull up script `msrun_launcher.sh` is located in the `scripts/` directory and can automatically start distributed multiprocess tasks using the [msrun](https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html) command based on the input parameters. This script has the following several usage methods:

1. For Default 8 Devices In Single Machine：

```bash
bash msrun_launcher.sh [EXECUTE_ORDER]
```

2. For Quick Start On Multiple Devices In Single Machine：

```bash
bash msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM]
```

3. For Multiple Devices In Single Machine：

```bash
bash msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM] [MASTER_PORT] [LOG_DIR] [JOIN] [CLUSTER_TIME_OUT]
```

4. For Multiple Devices In Multiple Machines：

```bash
bash msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM] [LOCAL_WORKER] [MASTER_ADDR] [MASTER_PORT] [NODE_RANK] [LOG_DIR] [JOIN] [CLUSTER_TIME_OUT]
```

The parameter descriptions of the script are as follows:

|     Parameters     | Parameter Descriptions                                                               | Value Description                                                                                       |
|:------------------:|:-------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
|  `EXECUTE_ORDER`   | The parameters of the Python script command to be executed in a distributed manner.  | str, required, set it to a string containing the Python script to be executed and the script parameters |
|    `WORKER_NUM`    | The total number of Worker processes participating in the distributed task.          | int, optional, default: `8`                                                                             |
|   `LOCAL_WORKER`   | The number of Worker processes pulled up on the current node.                        | int, optional, default: `8`                                                                             |
|   `MASTER_ADDR`    | Specifies the IP address or hostname of the Scheduler.                               | str, optional, default: `"127.0.0.1"`                                                                   |
|   `MASTER_PORT`    | Specifies the Scheduler binding port number.                                         | int, optional, default: `8118`                                                                          |
|    `NODE_RANK`     | The index of the current node.                                                       | int, optional, default: `0`                                                                             |
|     `LOG_DIR`      | Worker, and Scheduler log output paths.                                              | str, optional, default: `"output/msrun_log"`                                                            |
|       `JOIN`       | Whether msrun waits for the Worker as well as the Scheduler to exit.                 | bool, optional, default: `False`                                                                        |
| `CLUSTER_TIME_OUT` | Cluster networking timeout in seconds.                                               | int, optional, default: `7200`                                                                          |

## Task Startup Tutorial

Next, taking the fine-tuning of Qwen2.5-0.5B as an example, we will explain the usage of single-device, single-node, and multi-node tasks.

### Single-Device

Execute the Python script in the root directory of the MindSpore Transformers code to perform single-device fine-tuning. The path in the command needs to be replaced with the real path.

```shell
python run_mindformer.py \
--register_path research/qwen2_5 \
--config research/qwen2_5/finetune_qwen2_5_0_5b_8k.yaml \
--use_parallel False \
--run_mode finetune \
--train_dataset_dir ./path/alpaca-data.mindrecord
```

### Single-Node

Execute the msrun startup script in the root directory of the MindSpore Transformers code to perform single-node fine-tuning. The path in the command needs to be replaced with the real path.

```shell
bash scripts/msrun_launcher.sh "run_mindformer.py \
 --register_path research/qwen2_5 \
 --config research/qwen2_5/finetune_qwen2_5_0_5b_8k.yaml \
 --run_mode finetune \
 --train_dataset_dir ./path/alpaca-data.mindrecord "
```

### Multi-Node

Take Qwen2.5-0.5B as an example to perform 2-node 16-device fine-tuning.

1. Modify the corresponding config file `research/qwen2_5/finetune_qwen2_5_0_5b_8k.yaml` based on information such as the number of used nodes:

    ```yaml
    parallel_config:
      data_parallel: 2
      model_parallel: 4
      pipeline_stage: 2
      micro_batch_num: 16
      vocab_emb_dp: True
      gradient_aggregation_group: 4
    ```

    > If the number of nodes and the number of devices are used to change, `data_parallel`, `model_parallel`, and `pipeline_stage` need to be modified to meet the actual number of running devices . `device_num=data_parallel×model_parallel×pipeline_stage`. Meanwhile, `micro_batch_num >= pipeline_stage`.

2. Execute the msrun startup script:

    For distributed tasks by executing scripts on multiple nodes and multiple devices, it is necessary to run the scripts on different nodes respectively and set the parameter `MASTER_ADDR` to the ip address of the main node. The ip addresses set for all nodes are the same, and only the parameter `NODE_RANK` is different among different nodes. The meanings of each parameter position can be found in [msrun Launching](https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html).

    ```shell
    # Node 0. Set the IP address of node 0 to the value of {ip_addr}, which is used as the IP address of the primary node. There are 16 devices in total with 2 devices for each node.
    bash scripts/msrun_launcher.sh "run_mindformer.py \
      --register_path research/qwen2_5 \
      --config research/qwen2_5/finetune_qwen2_5_0_5b_8k.yaml \
      --train_dataset_dir /{path}/wiki4096.mindrecord \
      --run_mode finetune" \
      16 8 {ip_addr} 8118 0 output/msrun_log False 300


    # Node 1. Set the IP address of node 0 to the value of {ip_addr}, which is used as the IP address of the primary node. The startup commands of node 0 and node 1 differ only in the parameter NODE_RANK.
    bash scripts/msrun_launcher.sh "run_mindformer.py \
      --register_path research/qwen2_5 \
      --config research/qwen2_5/finetune_qwen2_5_0_5b_8k.yaml \
      --train_dataset_dir /{path}/wiki4096.mindrecord \
      --run_mode finetune" \
      16 8 {ip_addr} 8118 1 output/msrun_log False 300
    ```
