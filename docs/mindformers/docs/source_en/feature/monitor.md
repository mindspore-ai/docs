# Training Metric Monitoring and Profiling

The MindSpore Transformers dynamic graph (PyNative) provides two types of observability capabilities in the training loop:

- **Training metric monitoring** (`monitor`): collects fine-grained metrics such as gradient/norm and loss by micro-step (including the reserved configuration for MoE expert load monitoring). These metrics are currently **output only through training logs (`logger.info`)**.
- **Performance data collection** (`profiler`): collects performance data such as operators, communication, memory, and call stacks within a specified step range. The results are flushed to the disk as MindSpore Profiler products for performance analysis and optimization.

Both capabilities are enabled through the training YAML configuration and are independent of each other. They can be used separately. If you have not performed a training task, you are advised to read [Quick Start](../quick_start/quick_start.md) first. For details about the overall structure of the configuration file, see [Configuration File Description](./configuration.md).

> **Output Description**
>
> Currently, dynamic graph monitoring metrics (in the `monitor` section) **are written only to logs**. Each record is printed by `Monitor._flush_logger` in `{ key: value, ... }` format to the training log. To visualize the norm/loss curve, you need to parse the logs.

## 1 Training Metric Monitoring

The `monitor` section has two sub-configurations: `train_state` and `moe_monitor`.

The monitoring is instantiated by `MonitorGroup` during training initialization based on the configuration. Only the activated sub-monitoring is created: `train_state` is activated when any norm or loss switch is turned on; `moe_monitor` is activated when `save_tokens_per_expert_interval` is a positive integer. Metrics are flushed to logs by `MonitorCallback` (`on_step_end`) at the end of each training step.

### 1.1 train_state (Monitoring norm and loss)

The gradient norms and losses at the local and device levels are collected by micro-step. The four switches are independent of each other and can be combined in any way.

| Parameter         | Data Type                  | Required/Optional| Default Value     | Description                                                                                  |
|---------------|------------------------|------|----------|----------------------------------------------------------------------------------------|
| `local_norm`  | Boolean/String/list[str]| Optional  | `""`     | Specifies whether to monitor the local gradient norm. `True` indicates that all trainable parameters are monitored. A string or multiple strings in a list are used to filter parameters by parameter name substring (`"layers.0"` matches parameters whose names contain the substring). `""` or `False` indicates that this function is disabled.|
| `local_loss`  | Boolean                  | Optional  | `False`  | Specifies whether to monitor the local loss of each micro-batch.                                                               |
| `device_norm` | Boolean/String/list[str]| Optional  | `False`  | Specifies whether to monitor the device-level (accumulated) gradient norm. The value meaning is the same as that of `local_norm`.                                                     |
| `device_loss` | Boolean                  | Optional  | `False`  | Specifies whether to monitor the device-level loss (loss after gradient accumulation/all-reduce).                                                       |

> **device_norm forcibly collects local_norm.**
>
> In the source code, the accumulated norm of `device_norm` comes from the `prev_local_norms` cached during `local_norm` collection. Therefore, as long as `device_norm` is enabled, the framework computes the norm of the required parameters during the `local_norm` collection phase, even if `local_norm` itself is disabled. Both of them independently filter the parameters to be printed based on their parameter name filters.

#### Output Interpretation

- **`local_norm` is the incremental norm of this step.** The framework maintains the accumulated norm `accumulated` for each parameter and performs differentiation by micro-step:
    - micro-step 0: `actual_norm = accumulated`;
    - Subsequent steps: `actual_norm = accumulated – prev` (`prev` is the value of `accumulated` in the previous micro-step).
    In this way, each record reflects the **incremental contribution of the current micro-batch to the gradient norm**, rather than the total norm up to the current point.
- **`device_norm` is an accumulated value** and directly prints the accumulated norm cached in `prev_local_norms`.
- Logs are printed by parameter, for example, `{ "local_norm": decoder.layers.0.self_attention.q_layernorm.weight: 0.000381 }`. Before multiple norm records of the same micro-step are printed, a record in the format of `{ "step": 4, "micro_step": 2 }` is printed as a group marker.
- The `local_loss` record is in the format of `{ "micro_step": 2, "local_loss": 0.123 }`. `device_loss` is recorded after the optimizer is updated.

#### Applicable Scenarios

**Scenario A (Simplest): Viewing Only the Global Loss Curve**

If you are only concerned about the overall convergence and do not need to compute the norm of each parameter, **you do not need to enable any norm of train_state**. The step-level loss is directly printed to logs by `LossCallback` of the framework. The monitoring segment can be left empty or only `device_loss` can be enabled.

```yaml
monitor:
  train_state:
    device_loss: True       # Print the reduced loss after each step optimizer is updated.
```

**Scenario B: Checking for Gradient Explosion or Norm Exceptions at a Layer**

When a loss spike or NaN occurs, or you suspect that the gradient at some layers is abnormal, use `local_norm` to filter and locate the fault layer by parameter name. The incremental norm semantics (see "Output Interpretation" above) allow you to check **which micro-batch and which parameter** are used to contribute to the abnormal increment.

```yaml
monitor:
  train_state:
    # Only the incremental gradient norm of these types of parameters is monitored to avoid screen flooding by parameter.
    local_norm: ["attention", "layernorm"]
    local_loss: True        # Observe the local loss of each micro-batch to locate abnormal micro-batches.
```

> Note: To monitor all parameters, use `local_norm: True`. To monitor a single parameter, use a string, for example, `local_norm: "embedding"`.

**Scenario C: Observing Accumulation by Using device_norm and local_norm Together**

If you need to view both the incremental value per step and the total accumulated value, enable both of them. Note that `device_norm` reuses the collection result of `local_norm` (see the preceding note block).

```yaml
monitor:
  train_state:
    local_norm: True        # Incremental norm of each micro-step.
    device_norm: True       # Accumulated norm after gradient accumulation (local_norm collection is forcibly triggered).
    device_loss: True
```

### 1.2 moe_monitor (MoE Expert Load)

This function is designed to monitor the number of tokens (**tokens-per-expert**) allocated to each expert at the MoE decoding layer and is used to diagnose **uneven expert load**. If a few experts are saturated for a long time and the rest are idle, the routing is unbalanced. In this case, you need to adjust the load balancing loss or routing strategy.

| Parameter                             | Data Type                   | Required/Optional| Default Value    | Description                                                                                                                    |
|-----------------------------------|-------------------------|------|---------|--------------------------------------------------------------------------------------------------------------------------|
| `save_tokens_per_expert_interval` | int / None              | Optional  | `None`  | Prints the micro-step interval recorded by **tokens-per-expert**. You can set this parameter to `None` (indicating that the monitoring is disabled) or a positive integer. If you set this parameter to a Boolean value, an error will be reported.                                                   |
| `target_layers`                   | Integer/list[int]/None| Optional  | `None`  | Specifies the target layer to be monitored. `int` indicates the first *N* layers of `range(N)`, **matching only the decoder layer**. `list[int]` indicates the specified layer ID. `None` indicates that all MoE modules containing `tokens_per_expert` in the model are automatically discovered.|

> **Semantic differences between int and list of target_layers**
>
> - `int` (for example, `3`) is expanded to `range(3)`, that is, layers 0, 1, and 2 of the model.
> - `list[int]` (for example, `[0, 2, 4]`) is used to match these specified layer IDs.
> - If the layer with the specified ID is not an MoE layer (without `tokens_per_expert`), the alarm `decoder.layers.X is not a MoE layer` is reported in the framework.

#### Output Format

MoE monitoring information is printed to logs in the **tokens-per-expert JSON format of Megatron** (one JSON record per line). The fields are described as follows.

| Field          | Description                                         |
|--------------|---------------------------------------------|
| `iter`       | Global micro-step count (`global_micro_step`).      |
| `step`       | Training step ID.                                 |
| `micro_step` | Sequence number of micro-batch in a step.                     |
| `block`      | `"decoder"` or `"mtp"`.                      |
| `layer`      | Layer ID.                                       |
| `mtp_idx`    | MTP sublayer, which is displayed only for `block == "mtp"`.           |
| `tpe`        | List of tokens allocated to each expert in the micro-batch (increment obtained based on expert differentiation).|

#### Configuration Example: MoE Load Imbalance Diagnosis

The configuration of the monitoring is as follows:

```yaml
monitor:
  moe_monitor:
    save_tokens_per_expert_interval: 10   # Record every 10 micro-steps (design semantics).
    target_layers: [0, 5, 10]             # Sample the first, middle, and last layers to check for layer imbalance.
```

> You can also use `target_layers: 4` to monitor the first four layers, or omit this field to enable the framework to automatically discover all MoE layers (resulting in a larger data volume). The value of `save_tokens_per_expert_interval` must be a positive integer. If `True` or `False` is used, `TypeError` will be reported during configuration verification.

### 1.3 MaxLogits Health Monitoring (callbacks: MaxLogitsMonitor)

During large-scale long-term training, attention logits value overflow and gradient spikes can undermine training stability. `MaxLogitsMonitor` is a health monitoring callback for **the actual values consumed** in the dynamic graph. For the source code, see `mindformers/pynative/callback/max_logits_monitor.py`. Unlike the sub-configurations under the `monitor` section, it is registered through the **`callbacks` section**.

MaxLogitsMonitor collects the **maximum attention logit** of each layer of the model at each training step, prints the logit by layer to the log, and outputs the mean and maximum values of all layers. Then, it resets the accumulated value within each layer at the end of each step, facilitating the observation of whether the attention value increases abnormally step by step.

Output format (`_dump` and `max_logits_monitor.py:88-108`):

- One line per layer. The log tag is `max_attention_logit/<param_name>`, and the value is a list of items at the layer (four significant digits are retained).
- Two summary lines: `max_attention_logit/mean` and `max_attention_logit/max`.
- The prefix of each line is `step:[Current step/Total steps]`, which is the same as the printing format of `TrainingStateMonitor`.

#### Application Scenarios

- If you suspect that the loss jitter or NaN is caused by attention logits overflow or numerical explosion, use this function to locate the layer and step where the abnormal increase starts.
- When the Muon optimizer is used and the QK clip is enabled (in this case, the framework automatically enables tracing, as described in the following scenario), use this function to verify whether the clip takes effect.

#### Parameter Configuration

Register the callback through the `callbacks` section.

```yaml
callbacks:
  - type: MaxLogitsMonitor
    step_interval: 1   # Number of steps between outputs. The value must be a positive integer.
```

| Parameter            | Data Type| Required/Optional| Default Value | Description                                                                   |
|------------------|------|------|------|-------------------------------------------------------------------------|
| `step_interval`  | int  | Optional  | `1`  | Number of steps between outputs. The value must be a **positive integer**. Otherwise, `ValueError` (`max_logits_monitor.py:51-54`) is thrown during construction. |

Behavior details (`on_step_end` and `max_logits_monitor.py:58-78`):

- When `step_interval > 1` and the current step is not an output step, **only** the accumulated value **is reset** and is not printed.
- In the output step, `get_max_attention_logit()` of each layer is collected first. If no data is available, only the value is reset.
- Regardless of whether the value is printed, `reset_max_attention_logit()` is called at the end of each step to clear the value, ensuring that the next step starts from a clean state.

#### Scenario: Enabling the Automatic Tracking with Muon and QK Clip

`model.track_max_attention_logit` is used to specify whether to collect the max attention logit on the model side. The framework automatically specifies whether to enable this function in `configure_max_logits_tracking` (`max_logits_monitor.py:120-132`).

- The optimizer is **Muon** and `qk_clip_enabled` is `True`.
- `MaxLogitsMonitor` has been configured in `callbacks`.

When either of the preceding condition is met, `track_max_attention_logit` is set to `True`. This assembly is triggered during Trainer initialization (`pynative/trainer/trainer.py:153`), and ensures that a `MaxLogitsMonitor` is present in `ensure_max_logits_reset_callback` to handle the reset (`trainer.py:536-538`).

Therefore, in the **Muon + QK clip** scenario, even if no `MaxLogitsMonitor` is explicitly written, the framework automatically adds one for resetting. To view layer-by-layer logs, it is still recommended to explicitly write one and set `step_interval`.

```yaml
# Scenario: Explicitly enabling layer-by-layer max logit observation in Muon + QK clip
optimizer:
  type: Muon
  qk_clip_enabled: True       # Enable QK clip. The framework automatically enables track_max_attention_logit based on this setting.

callbacks:
  - type: MaxLogitsMonitor
    step_interval: 50         # Print the layer-by-layer max logit and mean/max every 50 steps.
```

> **Tracking (instead of printing) is automatically enabled.**
>
> For Muon + QK clip, only **model-side tracking** is automatically enabled, and a callback for resetting is added. To view the layer-by-layer numerical values in the logs, you still need to explicitly configure `MaxLogitsMonitor` (or accept the step-by-step output of the default `step_interval=1`).

## 2 Profile Data Collection

The `profiler` section controls profile data collection. The collection takes effect only when `enable_profiling: True` **and** the current rank hits `profiler_rank`. The framework uses MindSpore `schedule` to collect data within the `[start_step,end_step]` range. (Training outside the range is normal and no data is collected.)

| Parameter              | Data Type            | Required/Optional| Default Value    | Description                                                                                                                            |
|--------------------|------------------|------|---------|----------------------------------------------------------------------------------------------------------------------------------|
| `enable_profiling` | Boolean            | Optional  | `False` | Specifies whether to enable profile data collection.                                                                                                                     |
| `start_step`       | int              | Optional  | `1`     | Specifies the step at which collection starts (corresponding to `skip_first` of `schedule`). The value must be a positive integer.                                                                                 |
| `end_step`         | int              | Optional  | `1`     | Specifies the step at which collection ends. The value must be greater than or equal to `start_step` (the number of collected steps = `end_step – start_step + 1`). If a non-default `start_step` is set, `end_step` must also be set. Otherwise, the default `1` may cause an invalid range.|
| `output_path`      | String/None      | Optional  | `None`  | Specifies the path for saving the result. The result is saved in `output_path/rank_x` by rank. If this parameter is set to `None`, the result is rolled back to save in `Current working directory/profile/rank_x`.                                                  |
| `profiler_rank`    | list[int]/None | Optional  | `None`  | Specifies the list of rank IDs for which collection is enabled. `None` indicates that all ranks are collected.                                                                                        |
| `profiler_level`   | int              | Optional  | `0`     | Specifies the collection level, which can be `0`, `1`, or `2`. A higher level indicates more detailed information. (The value is mapped to `ProfilerLevel.Level0/1/2`. If the value is invalid, `LevelNone` is used.)                                                 |
| `mstx`             | Boolean            | Optional  | `False` | Specifies whether to enable lightweight mstx dotting (transparently transmitted to `_ExperimentalConfig`) to insert lightweight markers on the timeline.                                                                        |
| `profile_memory`   | Boolean            | Optional  | `False` | Specifies whether to collect tensor memory data.                                                                                                               |
| `profile_cpu`      | Boolean            | Optional  | `True`  | Specifies whether to collect CPU profiling activities.                                                                                                          |
| `with_stack`       | Boolean            | Optional  | `True`  | Specifies whether to collect call stack data on the Python side.                                                                                                             |

> **Avoiding the first step for warm-up during range selection**
>
> The first step usually includes one-off overheads such as compilation and initialization, and the collected data does not represent the steady state. You are advised to set `start_step` to several steps later (for example, `start_step: 5`) and collect data for only a few steps (for example, 2 to 3 steps) to avoid warm-up and control the product size and overhead.
>
> **Overhead trade-off**
>
> `profile_memory: True` and `with_stack: True` significantly increase the collection overhead and product size, which may slow down the collected steps. Enable this option only when checking memory or call stack issues. It is recommended that `profile_memory` be disabled when analyzing the time consumption of pure operators.

### 2.1 Rank Selection in Multi-Device Scenarios

In multi-device training, you do not need to collect data from all devices because the data volume is large and the data is redundant. You can use `profiler_rank` to specify the rank to be collected.

- In single-device tuning, only `[0]` is collected.
- When checking for communication imbalance, you can collect data from different ranks in the same parallel group for comparison (for example, `[0, 1]`).
- For details about the mapping between ranks and parallel dimensions (dp/tp/pp/cp), see [Distributed Parallel Training](./parallel_training.md). Select representative ranks based on the mapping.

### 2.2 Output Directory and Viewing Method

- **Directory structure**: `output_path/rank_x/` if `output_path` is set; `<cwd>/profile/rank_x/` if `None` is used.

### 2.3 Scenario-based Configuration (Three Levels Based on profiler_level)

#### Scenario A: Lightweight Operator-Level Analysis (Level 0)

Only the operator time consumption distribution is viewed, and hotspot kernels are located, resulting in the minimum overhead. Disable the memory and call stack.

```yaml
profiler:
  enable_profiling: True
  start_step: 5            # Skip the first four warm-up steps.
  end_step: 7              # Perform three-step collection.
  profiler_rank: [0]
  profiler_level: 0
  profile_memory: False
  profile_cpu: False
  with_stack: False
  output_path: "./output/profile"
```

#### Scenario B: Communication Analysis (Level 1)

If you need to analyze the communication time and the overlap between compute and communication in distributed training, increase the level to Level 1 and perform comparison collection for multiple ranks as required.

```yaml
profiler:
  enable_profiling: True
  start_step: 5
  end_step: 7
  profiler_rank: [0, 1]    # Comparison within the same group to observe communication imbalance.
  profiler_level: 1
  mstx: True               # Lightweight dotting, facilitating alignment of key phases on the timeline.
  profile_memory: False
  with_stack: False
  output_path: "./output/profile"
```

#### Scenario C: Including memory and call stack (Level 2)

To check the peak graphics memory usage or attribute time consumption to the Python call stack, enable Level 2 and enable memory and call stack collection (which has the highest overhead). In this case, you are advised to collect data for only a single device and a small number of steps):

```yaml
profiler:
  enable_profiling: True
  start_step: 5
  end_step: 6              # Perform two-step collection to control the size of the product.
  profiler_rank: [0]
  profiler_level: 2
  profile_memory: True     # Collect tensor memory.
  with_stack: True         # Collect the Python call stack.
  output_path: "./output/profile"
```

## Related Documents

- Log parsing and structure: [Logs](./logging.md)
- Configuration file overview: [Configuration File Description](./configuration.md)
- Training task startup: [Starting Tasks](./start_task.md)
- Parallel dimension and rank selection (affecting `profiler_rank` and distributed collection): [Distributed Parallel Training](./parallel_training.md)
- Full training process (including training status monitoring): [Training Guide](../guide/training.md)
- First training task: [Quick Start](../quick_start/quick_start.md)
