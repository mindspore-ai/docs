# Environment Variable List

[![View Source](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/vllm_mindspore/docs/source_en/user_guide/environment_variables/environment_variables.md)

| Environment Variable | Required for Basic Scenarios | Function |
|----------------------|-----------------------------|----------|
| `export vLLM_MODEL_BACKEND=MINDFORMER_MODELS` | Running MindSpore Transformers models | Distinguishes between MindSpore Transformers and vLLM MindSpore native models (default: native models) |
| `export PYTHONPATH=/xxx/mindformers-dev/:$PYTHONPATH` | Running models in MindSpore Transformers  Research directory | MindSpore Transformers must be installed from source, as research directory code is not packaged into whl files |
| `export MINDFORMERS_MODEL_CONFIG=/xxx.yaml` | Running MindSpore Transformers models | Configuration file for MindSpore Transformers models |
| `export MS_JIT_MODULES="vllm_mindspore,research"` | Greater than v0.7.3 version | Specifies modules require JIT static compilation in static graph mode; corresponds to top-level module names in imports |
| `export GLOO_SOCKET_IFNAME=enp189s0f0` | Ray multi-machine | Used for inter-server communication in Ray multi-machine scenarios |
| `export TP_SOCKET_IFNAME=enp189s0f0` | Ray multi-machine | Required for RPC in Ray multi-machine scenarios |
| `export HCCL_OP_EXPANSION_MODE=AIV` | Multi-machine | Multi-machine optimization configuring communication algorithm orchestration for acceleration |
| `export HCCL_EXEC_TIMEOUT=7200` | Multi-machine | Multi-machine optimization controlling device synchronization timeout (seconds, default: 1836) |
| `export RUN_MODE="predict"` | Basic inference workflow (system default) | Configures network execution mode (predict mode enables optimizations) |
| `export DEVICE_NUM_PER_NODE=16` | Multi-machine checkpoint splitting | Required for automatic weight splitting functionality (default: 8 NPUs/server) |
| `export vLLM_USE_NPU_ADV_STEP_FLASH_OP="on"` | MSS (Multi-step scheduler) custom operators | Toggle for custom operators in MSS functionality |
| `export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` | Ray multi-machine | Enables Ray dependency in vLLM MindSpore |
| `export MS_JIT=0` | Quantization scenarios (post v0.7.3) | 0: Disables JIT compilation, executing network scripts in dynamic graph (PyNative) mode |
| `export FORCE_EAGER="true"` | Quantization scenarios (post v0.7.3) |  |
