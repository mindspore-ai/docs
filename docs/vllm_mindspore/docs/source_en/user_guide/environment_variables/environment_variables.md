# Environment Variable List

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.0/docs/vllm_mindspore/docs/source_en/user_guide/environment_variables/environment_variables.md)

| Environment Variable | Function | Type | Values | Description |
|----------------------|----------|------|--------|-------------|
| `vLLM_MODEL_BACKEND` | Specifies the model backend. Not Required when using vLLM MindSpore native models, and required when using an external vLLM MindSpore models. | String | `MindFormers`: Model source is MindSpore Transformers. | vLLM MindSpore native model backend supports Qwen2.5 series. MindSpore Transformers model backend supports Qwen/DeepSeek/Llama series models, and the environment variable: `export PYTHONPATH=/path/to/mindformers/:$PYTHONPATH` needs to be set. |
| `MINDFORMERS_MODEL_CONFIG` | Configuration file for MindSpore Transformers models. Required for Qwen2.5 series or DeepSeek series models. | String | Path to the model configuration file | **This environment variable will be removed in future versions.** Example: `export MINDFORMERS_MODEL_CONFIG=/path/to/research/deepseek3/deepseek_r1_671b/predict_deepseek_r1_671b_w8a8.yaml`. |
| `GLOO_SOCKET_IFNAME` | Specifies the network interface name for inter-machine communication using gloo. | String | Interface name (e.g., `enp189s0f0`). | Used in multi-machine scenarios. The interface name can be found via `ifconfig` by matching the IP address. |
| `TP_SOCKET_IFNAME` | Specifies the network interface name for inter-machine communication using TP. | String | Interface name (e.g., `enp189s0f0`). | Used in multi-machine scenarios. The interface name can be found via `ifconfig` by matching the IP address. |
| `HCCL_SOCKET_IFNAME` | Specifies the network interface name for inter-machine communication using HCCL. | String | Interface name (e.g., `enp189s0f0`). | Used in multi-machine scenarios. The interface name can be found via `ifconfig` by matching the IP address. |
| `ASCEND_RT_VISIBLE_DEVICES` | Specifies which devices are visible to the current process, supporting one or multiple Device IDs. | String | Device IDs as a comma-separated string (e.g., `"0,1,2,3,4,5,6,7"`). | Recommended for Ray usage scenarios. |
| `HCCL_BUFFSIZE` | Controls the buffer size for data sharing between two NPUs. | int | Buffer size in MB (e.g., `2048`). | Usage reference: [HCCL_BUFFSIZE](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/maintenref/envvar/envref_07_0080.html). Example: For DeepSeek hybrid parallelism (Data Parallel: 32, Expert Parallel: 32) with `max-num-batched-tokens=256`, set `export HCCL_BUFFSIZE=2048`. |
| `MS_MEMPOOL_BLOCK_SIZE` | Set the size of the memory pool block in PyNative mode for devices | String | String of positive number, and the unit is GB. |  |
| `vLLM_USE_NPU_ADV_STEP_FLASH_OP` | Whether to use Ascend operation `adv_step_flash`  | String | `on`: Use；`off`：Not use | If the variable is set to `off`, model will use the implement of small operations. |
| `VLLM_TORCH_PROFILER_DIR` | Enables profiling data collection and takes effect when a data save path is configured. | String | The path to save profiling data. | |

The following environment variables are automatically registered by vLLM MindSpore:  

| **Environment Variable** | **Function** | **Type** | **Values** | **Description** |  
|------------------------|-------------|----------|------------|----------------|  
| `USE_TORCH` | Transformer runtime depends on this variable. | String | Default: `"False"` | vLLM MindSpore does not use Torch as the backend. |  
| `USE_TF` | Transformer runtime depends on this variable. | String | Default: `"False"` | vLLM MindSpore does not use TensorFlow as the backend. |  
| `RUN_MODE` | Execution mode. | String | Default: `"predict"` | **This variable will be removed in future versions.** Required by MindFormers. |  
| `CUSTOM_MATMUL_SHUFFLE` | Enables or disables custom matrix shuffling algorithm . | String | `on`: Enable shuffling. `off`: Disable shuffling. Default: `on`. | |  
| `HCCL_DETERMINISTIC` | Enables or disables deterministic computation for reduction-type communication operators (e.g., AllReduce, ReduceScatter, Reduce). | String | `true`: Enable deterministic mode. `false`: Disable deterministic mode. Default: `false`. | |  
| `ASCEND_LAUNCH_BLOCKING` | Controls whether operators run in synchronous mode during training or online inference. | Integer | `1`: Force synchronous execution. `0`: Do not force synchronous execution. Default: `0`. | |  
| `TE_PARALLEL_COMPILER` | Maximum number of parallel compilation processes for operators. Parallel compilation is enabled if greater than 1. | Integer | Positive integer; Max = CPU cores * 80% / # of Ascend AI processors. Range: 1~32. Default: `0`. | |  
| `LCCL_DETERMINISTIC` | Controls whether LCCL deterministic AllReduce (ordered addition) is enabled. | Integer | `1`: Enable deterministic mode. `0`: Disable deterministic mode. Default: `0`. | |  
| `MS_ENABLE_GRACEFUL_EXIT` | Enables graceful process termination. | Integer | `1`: Enable graceful exit. `Other values`: Disable graceful exit. Default: `0`. | |  
| `CPU_AFFINITY` | Optimizes CPU core binding for MindSpore inference. | String | `True`: Enable core binding. `False`: Disable core binding. Default: `True`. | **This variable will be removed in future versions.** Replaced by `set_cpu_affinity` API. |  
| `MS_ENABLE_INTERNAL_BOOST` | Enables or disables MindSpore framework's internal acceleration. | String | `on`: Enable acceleration. `off`: Disable acceleration. Default: `on`. | |  
| `MS_ENABLE_LCCL` | Controls whether the LCCL communication library is used. | Integer | `1`: Enable. `0`: Disable. Default: `0`. | |  
| `HCCL_EXEC_TIMEOUT` | Controls the synchronization timeout for inter-device execution. | Integer | Range: (0, 17340] (seconds). Default: `7200`. | |  
| `DEVICE_NUM_PER_NODE` | Number of devices per node. | Integer | Default: `16`. | |  
| `HCCL_OP_EXPANSION_MODE` | Configures the expansion location for communication algorithms. | String | `AI_CPU`: Expands on AI CPU compute units. `AIV`: Expands on AI Vector Core compute units. Default: `AIV`. | |  
| `MS_JIT_MODULES` | Specifies modules to be JIT-compiled in static graph mode. | String | Module names (top-level imports). Multiple modules should be comma-separated. Default: `"vllm_mindspore,research"`. | |  
| `GLOG_v` | Controls log level. | Integer | `0`: DEBUG. `1`: INFO. `2`: WARNING. `3`: ERROR (logs errors, may not terminate). `4`: CRITICAL (logs critical errors, terminates execution). Default: `3`. | |  
| `RAY_CGRAPH_get_timeout` | Timeout for `ray.get()` method (seconds). | Integer | Default: `360`. | |  
| `MS_NODE_TIMEOUT` | Node heartbeat timeout (seconds). | Integer | Default: `180`. | |  

More environment variable information can be referred in the following link:

 - [CANN Environment Variable List](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/81RC1beta1/index/index.html)
 - [MindSpore Environment Variable List](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/env_var_list.html)
 - [MindSpore Transformers Environment Variable List](https://www.mindspore.cn/mindformers/docs/en/r1.6.0/index.html)
 - [vLLM Environment Variable List](https://docs.vllm.ai/en/v0.8.4/serving/env_vars.html)
