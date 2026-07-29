# Starting Tasks

[![View Source on AtomGit](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.10.0/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/r2.10.0/docs/mindformers/docs/source_en/feature/start_task.md)

MindSpore Transformers dynamic graph (PyNative) training provides the one-click startup script `run_mindformer.py` and the distributed task startup script `msrun_launcher.sh`.

- The `run_mindformer.py` script is used to start a task on a **single device**. It provides the capability of starting a pretrain task in one click.
- The `msrun_launcher.sh` script is used to start distributed tasks on **single-node multi-device** or **multi-node multi-device** deployment. It uses the [msrun](https://www.mindspore.cn/tutorials/en/r2.10.0/parallel/msrun_launcher.html) tool to start tasks on each device.

## 1 One-Click Startup Script run_mindformer

In the root directory of the MindSpore Transformers code, use Python to execute the `run_mindformer.py` script to start the task. The script supports the following parameters. **If an optional parameter is not set or is set to `None`, the configuration with the same name in the YAML configuration file is used.**

### Basic Parameters

|       Parameter      | Data Type| Required/Optional| Default Value      | Description                                                                                                                       |
|:----------------:|:----:|:----:|:----------|:----------------------------------------------------------------------------------------------------------------------------|
|    `--config`    | String |  Required | None        | Path of the YAML configuration file of a task.                                                                                                             |
|     `--mode`     | int  |  Required | `1`       | Backend execution mode. You need to specify `1` to use PYNATIVE_MODE.                                                                                         |
|   `--run_mode`   | String |  Optional | Obtained from the YAML configuration. | Running mode of the model. The value can be `train`.                                                                                                      |
| `--use_parallel` | Boolean|  Optional | Obtained from the YAML configuration. | Specifies whether to enable the parallel mode.                                                                                                                  |
|  `--output_dir`  | String |  Optional | Obtained from the YAML configuration. | Path for storing files such as logs, weights, and sharding strategies.                                                                                                      |
|    `--seed`      | int  | Optional  | Obtained from the YAML configuration. | Global seed. For details, see [mindspore.set_seed](https://www.mindspore.cn/docs/en/r2.10.0/api_python/mindspore/mindspore.set_seed.html). |

## 2 Distributed Task Startup Script

The distributed task startup script `msrun_launcher.sh` is stored in the `scripts/` directory. It can automatically use the [msrun](https://www.mindspore.cn/tutorials/en/r2.10.0/parallel/msrun_launcher.html) command to start distributed multi-process tasks based on the input parameters. The script can be used in the following ways:

1. By default, 8 devices are running on a single node.

    ```bash
    bash scripts/msrun_launcher.sh [EXECUTE_ORDER]
    ```

2. Only a specified number of devices are running quickly on a single node.

    ```bash
    bash scripts/msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM]
    ```

3. Custom running is performed on a single node.

    ```bash
    bash scripts/msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM] [MASTER_PORT] [LOG_DIR] [JOIN] [CLUSTER_TIME_OUT]
    ```

4. Custom running is performed on multiple nodes.

    ```bash
    bash scripts/msrun_launcher.sh [EXECUTE_ORDER] [WORKER_NUM] [LOCAL_WORKER] [MASTER_ADDR] [MASTER_PORT] [NODE_RANK] [LOG_DIR] [JOIN] [CLUSTER_TIME_OUT]
    ```

The parameters of the script are described as follows.

|        Parameter        | Data Type| Required/Optional|         Default Value         | Description                         |
|:-------------------:|:----:|:----:|:--------------------:|:------------------------------|
|   `EXECUTE_ORDER`   | String |  Required |          None          | Python script command parameter to be executed in distributed mode.         |
|    `WORKER_NUM`     | int  |  Optional |         `8`          | Total number of worker processes that should be started on each node.          |
|   `LOCAL_WORKER`    | int  |  Optional |         `8`          | Number of worker processes started on the current node.           |
|    `MASTER_ADDR`    | String |  Optional |    `"127.0.0.1"`     | IP address or host name of the scheduler.       |
|    `MASTER_PORT`    | int  |  Optional |        `8118`        | Port number bound to the scheduler.            |
|     `NODE_RANK`     | int  |  Optional |         `0`          | Index of the current node.                     |
|      `LOG_DIR`      | String |  Optional | `"output/msrun_log"` | Output path of the worker and scheduler logs.     |
|       `JOIN`        | Boolean|  Optional |       `False`        | Specifies whether msrun waits for the worker and scheduler to exit.|
| `CLUSTER_TIME_OUT`  | int  | Optional  |       `7200`         | Timeout interval of the cluster network, in seconds.               |

## 3 Task Startup

The following uses the dynamic graph scenario as an example to describe how to use single-device, single-node, and multi-node tasks.

### Single-Device Startup

You can use the environment variable `ASCEND_RT_VISIBLE_DEVICES` to select a specific device to execute the task.

```bash
export ASCEND_RT_VISIBLE_DEVICES=7 # Select device 7 to execute the task.
```

> If no device is specified using environment variables, device 0 is selected by default to execute the single-device task.

Execute the Python script in the root directory of the MindSpore Transformers code to perform single-device training. The following is an example of the single-device startup command:

```bash
python run_mindformer.py --config /path/to/your.yaml --mode 1
```

> `ASCEND_RT_VISIBLE_DEVICES=0` indicates that only physical device 0 is visible to the process. To use device 3, write `=3`. For multiple devices, you can use a list format such as `=0,1,2,3`.

### Single-Node Multi-Device Startup

Execute the msrun startup script in the root directory of the MindSpore Transformers code to perform single-node training.

The parallel sharding dimension (such as FSDP/HSDP, TP, CP, PP, and EP) is determined by the `parallelism` section in the configuration file. Take the single-node 8-device scenario as an example. If the sharding is set to fsdp8, the corresponding YAML file fields are as follows:

```yaml
training:                    # Training hyperparameters (the naming style is torchtitan).
  steps: 10                  # Total number of training steps.
  local_batch_size: 1        # Batch size of each device (per rank).
  global_batch_size: 8       # Global batch size per step = Sum of DP shards (used to compute the number of gradient accumulation steps).
  seed: 42

parallelism:                 # Parallel sharding.
  data_parallel_shard: -1    # -1 indicates that the FSDP shard size is automatically derived based on the number of remaining devices.
  data_parallel_shard_strategy: "optim_grads_params"
  tensor_parallel: 1         # TP
  context_parallel: 1        # CP
  pipeline_parallel: 1       # PP
  expert_parallel: 1         # EP (not included in the product of world_size. For details, see the following description.)

train_dataset:
  dataloader:
    type: BlendedMegatronDatasetDataLoader
    datasets_type: "GPTDataset"
    sizes: [1000, 0, 0]
    column_names: ["input_ids", "labels", "loss_mask", "position_ids"]
    shuffle: false
    config:                    # For details about the complete fields, see the document "Datasets."
      seq_length: 4096
      split: "1, 0, 0"
      eod: 1
      pad: -1
      create_attention_mask: False
      create_compressed_eod_mask: False
      data_path:
        - '1'
        - "/path/megatron_data_text_document"
  drop_remainder: True
  num_parallel_workers: 8
```

After configuring the YAML file, use the `msrun_launcher.sh` script to start the task.

```bash
bash scripts/msrun_launcher.sh "run_mindformer.py \
--config /path/to/your.yaml \
--mode 1" 8
```

**Configuration sharding correspond to 8 devices**: The framework first computes the data parallelism degree and then shards it into `dp_replicate * dp_shard`:

- `data_parallel = world_size / (tensor_parallel * pipeline_parallel * context_parallel)` = `8 / (1*1*1)` = `8`
- When `data_parallel_shard = -1` is used, `dp_replicate = 1` and `dp_shard = data_parallel = 8` are automatically set, indicating that all 8 devices are used for FSDP sharding.
- Gradient accumulation steps `= global_batch_size/(data_parallel_shard * local_batch_size)` = `8/(8 * 1)` = `1`

> **Using world_size to verify that the degrees of parallelism multiplies together equal to the number of devices**
>
> The framework requires **`dp_replicate * dp_shard * cp * tp * pp == world_size`** (that is, `--worker_num`). If this requirement is not met, an error is reported.
>
> **`expert_parallel` (EP) is not included in this product** — EP is a reshard of experts within the DP/TP dimension and should not be multiplied into `world_size`. For example, when setting `expert_parallel: 2` on two devices, the configuration `dp_shard=2, tp=1, pp=1, cp=1` (Product = 2 = world_size) still passes the verification. EP only determines how experts are distributed between the two devices.

### Multi-Node Multi-Device Startup

In a multi-node scenario, **each node needs to execute the `msrun_launcher.sh` command**. Nodes are distinguished by `NODE_RANK` (0 for the primary node and 1 and 2 for secondary nodes). Other parameters must meet the following requirements:

- The values of `WORKER_NUM`, `MASTER_ADDR`, and `MASTER_PORT` **must be the same on all nodes**.
- Generally, the value of `LOCAL_WORKER` is also the same. **Set different values for each node only when the number of devices on each node is different.**
- The value of `NODE_RANK` **increases by node** to stably allocate rank IDs to each node.

The following uses a two-node system with 16 devices (8 devices per node) as an example. Assume that the IP address of the primary node is `192.168.1.1`.

**Primary node (node 0):**

```bash
# Node 0 serves as the primary node. There are a total of 16 devices, with 8 devices allocated per node.
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config /path/to/your.yaml \
  --mode 1" \
  16 8 192.168.1.1 8118 0 output/msrun_log False 7200
```

**Secondary node (node 1):**

```bash
# Node 1, with primary node IP address 192.168.1.1, has the same launch command as node 0, with the only difference being the NODE_RANK parameter.
bash scripts/msrun_launcher.sh "run_mindformer.py \
  --config /path/to/your.yaml \
  --mode 1" \
  16 8 192.168.1.1 8118 1 output/msrun_log False 7200
```

The two commands **only differ in the value of `NODE_RANK`** (`0` and `1`), with all other parameters being identical. If the number of devices on each node is different, you need to adjust `LOCAL_WORKER` of the corresponding node. However, `WORKER_NUM` must still be the total number of devices globally.

> **Risks of not setting NODE_RANK**
>
> If `NODE_RANK` is not set, MindSpore automatically allocates rank IDs (consecutive ranks on the same node). However, the rank IDs between nodes depend on the startup sequence of each node, which may lead to incorrect configuration and non-reproducibility. Therefore, **you must explicitly set `NODE_RANK` in multi-node scenarios**.

Before starting a multi-node task, ensure that the network between nodes is normal, `MASTER_PORT` is not occupied, and all nodes use **the same code, configuration, and dataset path**. For a large cluster, if the networking is slow, you can increase the value of `CLUSTER_TIME_OUT`.

## 4 Startup with a Custom Script

PyNative training allows you to directly build and start a trainer in a custom script. For example, you can write the following custom script and specify `config` as the corresponding YAML address to start PyNative training:

```python
from mindformers.pynative.trainer import Trainer as PynativeTrainer

def main():
    my_config = "/path/to/your.yaml"

    trainer = PynativeTrainer(config=my_config)
    trainer.train()

if __name__ == "__main__":
    main()
```

> **Differences between the unified entry of custom scripts and run_mindformer:**
>
> - Both `run_mindformer.py --mode 1` and custom scripts **build the same `PynativeTrainer` and call `train()`**. The running behavior is the same.
> - The only difference lies in who parses the command line and assembles the configuration. The unified entry is implemented by `run_mindformer.py`. The custom script is implemented by specifying `config` and other input parameters, which is more flexible and allows you to modify the configuration and perform tests in the script.
> - The test case in the repository (for example, `tests/st/test_multi_cards_cases/test_pynative/test_models/test_deepseek3/run_deepseek3.py`) is a custom script. Therefore, when you run the repository command, `bash scripts/msrun_launcher.sh "run_deepseek3.py --config xxx.yaml"` instead of `--mode 1` is displayed.

## 5 Troubleshooting During Startup

Most startup failures are related to networking, ports, and differences between single-device and multi-device scenarios. You can refer to the following table to quickly locate the fault.

| Symptom                       | Troubleshooting                                                                                                           |
|---------------------------|-----------------------------------------------------------------------------------------------------------------|
| The process is stuck in the initialization phase, and a message is displayed indicating that the number of workers is insufficient.| Not all nodes are started or the network is disconnected. Check whether `WORKER_NUM` of each node is consistent and whether `MASTER_ADDR/MASTER_PORT` is correct. For a large cluster, increase the value of `CLUSTER_TIME_OUT` (a negative value indicates infinite waiting).      |
| A message is displayed indicating that the port is occupied or the scheduler fails to be started.  | `MASTER_PORT` is occupied or is not within the range of `1024–65535`. Use an unoccupied port and ensure that all nodes use the same port.                                                 |
| Incorrect rank is configured in multi-node training and the training hangs in the communication phase. | `NODE_RANK` is not set. Explicitly set it in the sequence of primary node 0 and secondary node 1.                                                                         |
| HCCL/networking timeout occurs or the cross-node collective communication error is reported.       | Check the node connectivity and HCCL configuration. Increase the value of `CLUSTER_TIME_OUT`. Ensure that the configuration of each node is consistent with the data path.                                                       |
| A single device can run properly, but multiple devices fail to run.           | Most of the problems are caused by parallel configuration or collective communication. Check `dp_replicate * dp_shard * cp * tp * pp == world_size` (excluding the EP) first, and then check whether there are multi-node `NODE_RANK` or networking problems. |

**Viewing logs:** `msrun` generates an independent log file for each worker in the `LOG_DIR` directory (the scheduler and each worker are recorded separately). If an error is reported during the first startup, check the log of the corresponding worker instead of the aggregated output on the frontend. When `JOIN` is set to `True`, `msrun` parses the log and reclaims the exit code.

## Related Documents

- Configuration file fields and schemas: [Configuration File Description](./configuration.md)
- Parallelism dimension, world_size inference, and batch size: [Distributed Parallel Training](./parallel_training.md)
- End-to-end startup example: [Quick Start](../quick_start/quick_start.md)
