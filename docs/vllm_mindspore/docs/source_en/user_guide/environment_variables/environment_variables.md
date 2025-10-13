# Environment Variable List

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.7.1/docs/vllm_mindspore/docs/source_en/user_guide/environment_variables/environment_variables.md)

| Environment Variable | Function | Type | Values | Description |
|----------------------|----------|------|--------|-------------|
| `VLLM_MS_MODEL_BACKEND` | Used to specify the model backend. If this variable is not set, the backend will be automatically selected in the priority order: MindFormers > Native > MindONE; if set, the specified backend will be used. | String | `MindFormers`: Model backend is MindSpore Transformers. `Native`: Model backend is Native. `MindONE`: Model backend is MindONE. | The native model backend currently supports the Qwen2.5, Qwen2.5VL, Qwen3 and Llama series; the MindSpore Transformers backend supports Qwen, DeepSeek and TeleChat models. |
| `GLOO_SOCKET_IFNAME` | Specifies the network interface name for inter-machine communication using gloo. | String | Interface name (e.g., `enp189s0f0`). | Used in multi-machine scenarios. The interface name can be found via `ifconfig` by matching the IP address. |
| `TP_SOCKET_IFNAME` | Specifies the network interface name for inter-machine communication using TP. | String | Interface name (e.g., `enp189s0f0`). | Used in multi-machine scenarios. The interface name can be found via `ifconfig` by matching the IP address. |
| `HCCL_SOCKET_IFNAME` | Specifies the network interface name for inter-machine communication using HCCL. | String | Interface name (e.g., `enp189s0f0`). | Used in multi-machine scenarios. The interface name can be found via `ifconfig` by matching the IP address. |
| `ASCEND_RT_VISIBLE_DEVICES` | Specifies which devices are visible to the current process, supporting one or multiple Device IDs. | String | Device IDs as a comma-separated string (e.g., `"0,1,2,3,4,5,6,7"`). | Recommended for Ray usage scenarios. |
| `HCCL_BUFFSIZE` | Controls the buffer size for data sharing between two NPUs. | int | Buffer size in MB (e.g., `2048`). | Usage reference: [HCCL_BUFFSIZE](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1beta1/maintenref/envvar/envref_07_0080.html). Example: For DeepSeek hybrid parallelism (Data Parallel: 32, Expert Parallel: 32) with `max-num-batched-tokens=256`, set `export HCCL_BUFFSIZE=2048`. |
| `MS_MEMPOOL_BLOCK_SIZE` | Set the size of the memory pool block in PyNative mode for devices | String | String of positive number, and the unit is GB. |  |
| `vLLM_USE_NPU_ADV_STEP_FLASH_OP` | Whether to use Ascend operation `adv_step_flash`  | String | `on`: Use; `off`: Not use | If the variable is set to `off`, model will use the implementation of small operations. |
| `VLLM_TORCH_PROFILER_DIR` | Enables profiling data collection and takes effect when a data save path is configured. | String | The path to save profiling data. | |

More environment variable information can be referred in the following links:

- [CANN Environment Variable List](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/81RC1beta1/index/index.html)
- [MindSpore Environment Variable List](https://www.mindspore.cn/docs/en/r2.7.1/api_python/env_var_list.html)
- [MindSpore Transformers Environment Variable List](https://www.mindspore.cn/mindformers/docs/en/r1.7.0/index.html)
- [vLLM Environment Variable List](https://docs.vllm.ai/en/v0.8.4/serving/env_vars.html)
