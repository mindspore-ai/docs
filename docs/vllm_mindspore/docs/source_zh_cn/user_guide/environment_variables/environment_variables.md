# 环境变量清单

|   环境变量   |   必配基础场景   |   功能   |
|   ------   |   ----------  |   -------  |
|   export vLLM_MODEL_BACKEND=MINDFORMER_MODELS   |   运行mindformers模型   |   用于区分mindformers和vllm-mindspore原生模型，默认原生模型   |
|   export PYTHONPATH=/xxx/mindformers-dev/:$PYTHONPATH   |   运行mindformers的research下模型   |   mindformers要用源码安装，因为research目录下代码不打包到whl中   |
|   export MINDFORMERS_MODEL_CONFIG=/xxx.yaml   |   运行mindformers模型   |   mindformers模型的必须配置文件   |
|   export MS_JIT_MODULES="vllm_mindspore,research"   |   升级0.7.3后版本   |   指定静态图模式下哪些模块需要JIT静态编译，其函数方法会被编译成静态计算图; 对应import导入的顶层模块的名称   |
|   export GLOO_SOCKET_IFNAME=enp189s0f0   |   Ray多机   |   Ray多机场景使用，用于服务器间通信   |
|   export TP_SOCKET_IFNAME=enp189s0f0   |   Ray多机   |   Ray多机场景使用，RPC时需要设置   |
|   export HCCL_OP_EXPANSION_MODE=AIV   |   多机   |   多机场景优化，配置通信算法的编排展开位置，用于通信加速   |
|   export HCCL_EXEC_TIMEOUT=7200   |   多机   |   多机场景优化，控制设备间执行时同步等待的时间，单位为s，默认值为1836   |
|   export RUN_MODE="predict"   |   推理基础流程---系统默认配置   |   配置网络执行模式，predict模式下会使能一些优化   |
|   export DEVICE_NUM_PER_NODE=16   |   多机使用ckpt切分   |   自动权重切分要识别卡数功能依赖，单机实际NPU数量，不设置默认为8卡服务器   |
|   export vLLM_USE_NPU_ADV_STEP_FLASH_OP="on"   |   mss（Multi step scheduler）自定义算子   |   mss（Multi step scheduler）功能中自定义算子开关   |
|   export ASCEND_RT_VISIBLE_DEVICES=0,1   |   vllm-ascend Ray多机场景   |   vllm-ascend中使能Ray依赖   |
|   export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1   |   vllm-ascend Ray多机场景   |   vllm-ascend中使能Ray依赖   |
|   export MS_JIT=0   |   量化场景，升级0.7.3后版本   |   0：不使用JIT即时编译，网络脚本直接按照动态图（PyNative）模式执行。   |
|   export FORCE_EAGER="true"   |   量化场景，升级0.7.3后版本   |       |
