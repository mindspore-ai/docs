# Ascend转换工具功能说明

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_zh_cn/mindir/converter_tool_ascend.md)

## 概述

本文档介绍云侧推理模型转换工具在Ascend后端的相关功能，如配置文件的选项、动态shape、AOE、自定义算子等。

## 配置文件

表1：配置ascend_context参数

| 参数                        | 属性  | 功能描述                                                       | 参数类型 | 取值说明 |
| -------------------------- | ---- | ------------------------------------------------------------ | -------- | ------ |
| `input_format`             | 可选 | 指定模型输入format。 | String | 可选有`"NCHW"`、`"NHWC"`、`"ND"` |
| `input_shape`       | 可选 | 指定模型输入Shape，input_name必须是转换前的网络模型中的输入名称，按输入次序排列，用`；`隔开，仅对动态BatchSize生效，对静态BatchSize，需要converter_lite命令指定inputShape参数。 | String | 例如：`"input1:[1,64,64,3];input2:[1,256,256,3]"` |
| `dynamic_dims`       | 可选 | 指定动态BatchSize和动态分辨率参数。 | String | 见[动态shape配置](#动态shape配置) |
| `precision_mode`           | 可选 | 配置模型精度模式。    | String | 可选有`"enforce_fp32"`，`"preferred_fp32"`，`"enforce_fp16"`，`"enforce_origin"`或者`"preferred_optimal"`，默认为`"enforce_fp16"`|
| `op_select_impl_mode`      | 可选 | 配置算子选择模式。    | String | 可选有`"high_performance"`和`"high_precision"`，默认为`"high_performance"` |
| `output_type`       | 可选 | 指定网络输出数据类型。  | String | 可选有`"FP16"`、`"FP32"`、`"UINT8"` |
| `fusion_switch_config_file_path` | 可选 | 配置[融合规则开关配置](https://www.hiascend.com/document/detail/zh/canncommercial/601/inferapplicationdev/atctool/atctool_0078.html)文件路径及文件名。 | String   | 指定融合规则开关的配置文件      |
| `insert_op_config_file_path` | 可选 | 模型插入[AIPP](https://www.hiascend.com/document/detail/zh/canncommercial/601/inferapplicationdev/atctool/atctool_0018.html)算子 | String  | [AIPP](https://www.hiascend.com/document/detail/zh/canncommercial/601/inferapplicationdev/atctool/atctool_0021.html)配置文件路径 |
| `aoe_mode` | 可选 | [AOE](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/800alpha001/migration/tfmigr1/tfmigr1_000052.html)自动调优模式 | String  | 可选有"subgraph tuning"、"operator tuning"或者"subgraph tuning、operator tuning"，默认不使能 |
| `plugin_custom_ops` | 可选 | 用于使能ascend后端融合优化生成自定义算子 | String  | 可选有`All`、`None`、`FlashAttention`、`LayerNormV3`、`GeGluV2`、`GroupNormSilu`、`FFN`、`AddLayerNorm`、`MatMulAllReduce`和`BatchMatmulToMatmul`，其中`All`表示使能`FlashAttention`、`LayerNormV3`、`GeGluV2`和`GroupNormSilu`，默认`None`表示不使能 |
| `custom_fusion_pattern` | 可选 | 指定使能模型中的自定义算子结构 | String  | `自定义算子类型:模型中原始算子名称:是否使能`，可以取值为`enable`或者`disable` |
| `op_attrs` | 可选 | 指定融合的自定义算子属性 | String | `自定义算子名:属性:值`，目前算子仅支持`FlashAttention`，该算子支持3种可选配置属性：`input_layout`、`seq_threshold`、`inner_precise`，分别决定`FlashAttention`以`BNSD`（默认）、`BSH`或`BNSD_BSND`（`BNSD`表示`FlashAttention`输入和输出的`layout`均是`BNSD`，`BSH`表示输入和输出均是`BSH`，`BNSD_BSND`表示输入是`BNSD`，输出为`BSND`）形式进行融合、融合`FlashAttention`的`seq`阈值（默认`0`）、高性能（默认）或高精度 |
| `stream_label_file` | 可选 | 指定算子多流配置文件路径 | String | 该配置文件按照 `流标签：算子名1,算子名2`的格式指定哪些算子在其他流执行，流标签为字符串，模型中默认已有一个流。该配置文件中配置的算子，会运行在新启动的流，如果希望起更多流，则可以配置多行并且每行配置不同的流标签。 |

表2：配置[acl_init_options]参数

| 参数                          |  属性  | 功能描述                                      | 参数类型 | 取值说明 |
| ---------------------------- | ----- | -------------------------------------------- | -------- | ------ |
| `ge.engineType`              |  可选  | 设置网络模型使用的Core类型。 | String | 可选有`"VectorCore"`、`"AiCore"` |
| `ge.socVersion`              |  可选  | 昇腾AI处理器的版本。 | String | 可选有`"Ascend310"`、`"Ascend710"`、`"Ascend910"` |
| `ge.bufferOptimize`          |  可选  | 数据缓存优化开关。 | String | 可选有`"l1_optimize"`、`"l2_optimize"`、`"off_optimize"`，默认为`"l2_optimize"` |
| `ge.enableCompressWeight`    |  可选  | 可以对Weight进行数据压缩，提升性能。 | String | 可选有`"true"`、`"false"` |
| `compress_weight_conf`       |  可选  | 要压缩的node节点列表配置文件路径，node节点主要为conv算子、fc算子。 | String | 配置文件路径 |
| `ge.exec.precision_mode`     |  可选  | 选择算子精度模式。 | String | 可选有`"force_fp32"`、`"force_fp16"`、`"allow_fp32_to_fp16"`、`"must_keep_origin_dtype"`、`"allow_mix_precision"`，默认为`"force_fp16"` |
| `ge.exec.disableReuseMemory` |  可选  | 内存复用开关。 | String | 可选有`"0"`、`"1"` |
| `ge.enableSingleStream`      |  可选  | 是否使能一个模型只使用一个stream。 | String | 可选有`"true"`、`"false"` |
| `ge.aicoreNum`               |  可选  | 设置编译时使用的ai core数目。 | String | 默认`"10"` |
| `ge.fusionSwitchFile`        |  可选  | 融合配置文件路径。 | String | 配置文件路径 |
| `ge.enableSmallChannel`      |  可选  | 是否使能small channel的优化。 | String | 可选有`"0"`、`"1"` |
| `ge.opSelectImplmode`        |  可选  | 选择算子实现模式。 | String | 可选有`"high_precision"`、`"high_performance"` |
| `ge.optypelistForImplmode`   |  可选  | 算子列表，列表中算子使用`ge.opSelectImplmode`参数指定的模式。 | String | 算子类型 |
| `ge.op_compiler_cache_mode`  |  可选  | 配置算子编译磁盘缓存模式。 | String | 可选有`"enable"`、`"force"`、`"disable"` |
| `ge.op_compiler_cache_dir`   |  可选  | 配置算子变异磁盘缓存目录。 | String | 默认值`$HOME/atc_data` |
| `ge.debugDir`                |  可选  | 配置保存算子编译生成的调试相关的过程文件的路径。 | String | 默认生成当前路径 |
| `ge.opDebugLevel`            |  可选  | 算子debug功能开关。 | String | 可选有`"0"`、`"1"` |
| `ge.exec.modify_mixlist`     |  可选  | 配置混合精度名单。 | String | 配置文件路径 |
| `ge.enableSparseMatrixWeight`|  可选  | 使能全局稀疏特性。 | String | 可选有`"1"`、`"0"` |
| `ge.externalWeight`          |  可选  | 是否将常量节点的权重单独保存到文件中。 | String | 可选有`"1"`、`"0"` |
| `ge.deterministic`           |  可选  | 是否开启确定性计算。 | String | 可选有`"1"`、`"0"` |
| `ge.host_env_os`             |  可选  | 支持编译环境操作系统与运行环境不一致。 | String | 可选有`"linux"` |
| `ge.host_env_cpu`            |  可选  | 支持编译环境操作系统架构与运行环境不一致。 | String | 可选有`"aarch64"`、`"x86_64"` |
| `ge.virtual_type`            |  可选  | 是否支持离线模型在昇腾虚拟化实例特性生成的虚拟设备上运行。 | String | 可选有`"0"`、`"1"` |
| `ge.compressionOptimizeConf` |  可选  | 压缩优化功能配置文件路径。 | String | 配置文件路径 |

表3：配置[acl_build_options]参数

| 参数                          | 属性  | 功能描述        | 参数类型 | 取值说明 |
| ----------------------------------- | ---- | ------------------------------------------------------------ | -------- | ------ |
| `input_format`                      | 可选 | 指定模型输入format。 | String | 可选有`"NCHW"`、`"NHWC"`、`"ND"` |
| `input_shape`                       | 可选 | 模型输入shape。模型转换后可以用Model.get_model_info("input_shpae")获取到。该参数与命令行中input_shape已统一。 | String | 例如：`input1:1,3,512,512;input2:1,3,224,224` |
| `op_name_map`                       | 可选 | 扩展算子映射配置文件路径。 | String | 配置文件路径 |
| `ge.dynamicBatchSize`               | 可选 | 设置动态batch档位参数。 | String | 该参数需要与`input_shape`参数配合使用 |
| `ge.dynamicImageSize`               | 可选 | 设置输入图片的动态分辨率参数。 | String | 该参数需要与`input_shape`参数配合使用 |
| `ge.dynamicDims`                    | 可选 | 设置ND格式下的动态维度的档位。模型转换后可以用Model.get_model_info("dynamic_dims")获取 | String | 该参数需要与`input_shape`参数配合使用 |
| `ge.inserOpFile`                    | 可选 | 输入预处理算子的配置文件路径。 | String | 配置文件路径 |
| `ge.exec.precision_mode`            | 可选 | 选择算子精度模式。 | String | 可选有`"force_fp32"`、`"force_fp16"`、`"allow_fp32_to_fp16"`、`"must_keep_origin_dtype"`、`"allow_mix_precision"`，默认为`"force_fp16"` |
| `ge.exec.disableReuseMemory`        | 可选 | 内存复用开关。 | String | 可选有`"0"`、`"1"` |
| `ge.outputDataType`                 | 可选 | 网络输出数据类型。 | String | 可选有`"FP32"`、`"UINT8"`、`"FP16"` |
| `ge.outputNodeName`                 | 可选 | 指定输出节点。 | String | 例如：`"node_name1:0;node_name1:1;node_name2:0"` |
| `ge.INPUT_NODES_SET_FP16`           | 可选 | 指定输入数据类型为FP16的输入节点名称。 | String | `"node_name1;node_name2"` |
| `log`                               | 可选 | 设置日志级别。 | String | 可选有`"debug"`、`"info"`、`"warning"`、`"error"` |
| `ge.op_compiler_cache_mode`         | 可选 | 配置算子编译磁盘缓存模式。 | String | 可选有`"enable"`、`"force"`、`"disable"` |
| `ge.op_compiler_cache_dir`          | 可选 | 配置算子变异磁盘缓存目录。 | String | 默认值`$HOME/atc_data` |
| `ge.debugDir`                       | 可选 | 配置保存算子编译生成的调试相关的过程文件的路径。 | String | 默认生成当前路径 |
| `ge.opDebugLevel`                   | 可选 | 算子debug功能开关。 | String | 可选有`"0"`、`"1"` |
| `ge.mdl_bank_path`                  | 可选 | 加载模型调优后自定义知识库的路径。 | String | 该参数需要和`ge.bufferOptimize`配合使用 |
| `ge.op_bank_path`                   | 可选 | 加载算子调优后自定义知识库路径。 | String | 知识库路径 |
| `ge.exec.modify_mixlist`            | 可选 | 配置混合精度名单。 | String | 配置文件路径 |
| `ge.exec.op_precision_mode`         | 可选 | 设置具体某个算子的精度模式，通过该参数设置配置文件路径。 | String | 配置文件路径 |
| `ge.shape_generalized_build_mode`   | 可选 | 图编译时shape编译方式。 | String | 可选有`"shape_generalized"`模糊编译、`"shape_precise"`精确编译 |
| `op_debug_config`                   | 可选 | 内存检测功能开关。 | String | 配置文件路径 |
| `ge.externalWeight`                 | 可选 | 是否将常量节点的权重单独保存到文件中。 | String | 可选有`"1"`、`"0"` |
| `ge.exec.exclude_engines`           | 可选 | 设置网络模型不使用某个或者某些加速引擎。 | String | 可选有`"AiCore"`、`"AiVec"`、`"AiCpu"` |

## 动态shape配置

在某些推理场景，如检测出目标后再执行目标识别网络，由于目标个数不固定导致目标识别网络输入BatchSize不固定。如果每次推理都按照最大的BatchSize或最大分辨率进行计算，会造成计算资源浪费。因此，推理需要支持动态BatchSize和动态分辨率的场景，Lite在Ascend上推理支持动态BatchSize和动态分辨率场景，在convert阶段通过configFile配置[ascend_context]中dynamic_dims动态参数，推理时使用model的[Resize](https://www.mindspore.cn/lite/docs/zh-CN/master/mindir/runtime_cpp.html#%E5%8A%A8%E6%80%81shape%E8%BE%93%E5%85%A5)功能，改变输入shape。

### 动态Batch size

- 参数名

    dynamic_dims

- 功能

    设置动态batch档位参数，适用于执行推理时，每次处理图片数量不固定的场景，该参数需要与input_shape配合使用，input_shape中-1的位置为动态batch所在的维度。

- 取值

    最多支持100档配置，每一档通过英文逗号分隔，每个档位数值限制为：[1~2048]。例如配置文件中参数配置如下：

    ```
    [ascend_context]
    input_shape=input:[-1,64,64,3]
    dynamic_dims=[1],[2]
    ```

    其中，input_shape中的"-1"表示设置动态batch，档位可取值为"1,2"，即支持档位0：[1,64,64,3]，档位1：[2,64,64,3]。

    若存在多个输入，不同输入对应的挡位需要一致，并用`;`隔开。

    ```
    [ascend_context]
    input_shape=input1:[-1,64,64,3];input2:[-1,256,256,3]
    dynamic_dims=[1],[2];[1],[2]
    ```

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --configFile=./config.txt --optimize=ascend_oriented --outputFile=${model_name}
    ```

    说明：使能动态BatchSize时，不需要指定inputShape参数，仅需要通过configFile配置[ascend_context]动态batch size，即上节示例中配置内容。

- 推理

    使能动态BatchSize，进行模型推理时，输入shape只能选择converter时设置的档位值，想切换到其他档位对应的输入shape，使用model [Resize](https://www.mindspore.cn/lite/docs/zh-CN/master/mindir/runtime_cpp.html#%E5%8A%A8%E6%80%81shape%E8%BE%93%E5%85%A5)功能。

- 注意事项

    1）若用户执行推理业务时，每次处理的图片数量不固定，则可以通过配置该参数来动态分配每次处理的图片数量。例如用户执行推理业务时需要每次处理2张、4张、8张图片，则可以配置为2,4,8，申请了档位后，模型推理时会根据实际档位申请内存。<br/>
    2）如果用户设置的档位数值过大或档位过多，可能会导致模型编译失败，此时建议用户减少档位或调低档位数值。<br/>
    3）如果用户设置的档位数值过大或档位过多，在运行环境执行推理时，建议执行swapoff -a命令关闭swap交换区间作为内存的功能，防止出现由于内存不足，将swap交换空间作为内存继续调用，导致运行环境异常缓慢的情况。<br/>

### 动态分辨率

- 参数名

    dynamic_dims

- 功能

    设置输入图片的动态分辨率参数。适用于执行推理时，每次处理图片宽和高不固定的场景，该参数需要与input_shape配合使用，input_shape中-1的位置为动态分辨率所在的维度。

- 取值

    最多支持100档配置，每一档通过英文逗号分隔。例如： "[imagesize1_height,imagesize1_width],[imagesize2_height,imagesize2_width]"。例如配置文件中参数配置如下：

    ```
    [ascend_context]
    input_format=NHWC
    input_shape=input:[1,-1,-1,3]
    dynamic_dims=[64,64],[19200,960]
    ```

    其中，input_shape中的"-1"表示设置动态分辨率，即支持档位0：[1,64,64,3]，档位1：[1,19200,960,3]。

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --configFile=./config.txt --optimize=ascend_oriented --outputFile=${model_name}
    ```

    说明：使能动态BatchSize时，不需要指定inputShape参数，仅需要通过configFile配置[ascend_context]动态分辨率，即上节示例中配置内容。

- 推理

    使能动态分辨率，进行模型推理时，输入shape只能选择converter时设置的档位值，想切换到其他档位对应的输入shape，使用model的[Resize](https://www.mindspore.cn/lite/docs/zh-CN/master/mindir/runtime_cpp.html#%E5%8A%A8%E6%80%81shape%E8%BE%93%E5%85%A5)功能。

- 注意事项

    1）如果用户设置的分辨率数值过大或档位过多，可能会导致模型编译失败，此时建议用户减少档位或调低档位数值。<br/>
    2）如果用户设置了动态分辨率，实际推理时，使用的数据集图片大小需要与具体使用的分辨率相匹配。<br/>
    3）如果用户设置的分辨率数值过大或档位过多，在运行环境执行推理时，建议执行swapoff -a命令关闭swap交换区间作为内存的功能，防止出现由于内存不足，将swap交换空间作为内存继续调用，导致运行环境异常缓慢的情况。<br/>

### 动态维度

- 参数名

    `ge.dynamicDims`

- 功能

    设置ND格式下输入的动态维度的档位。适用于执行推理时，每次处理任意维度的场景，该参数需要与`input_shape`配合使用，`input_shape`中-1的位置为动态维度。

- 取值

    最多支持100档配置，每一档通过英文逗号分隔。例如配置文件中参数配置如下：

    ```
    [acl_build_options]
    input_format="ND"
    input_shape="input1:1,-1,-1;input2:1,-1"
    ge.dynamicDims="32,32,24;64,64,36"
    ```

    其中，input_shape中的"-1"表示设置动态维度，即支持档位0：input1:1,32,32; input2:1,24，档位1：1,64,64; input2:1,36。

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --configFile=./config.txt --optimize=ascend_oriented --outputFile=${model_name}
    ```

    说明：使能动态维度时，`input_format`必须设置为`ND`。

- 推理

    使能动态维度，进行模型推理时，输入shape只能选择converter时设置的档位值，想切换到其他档位对应的输入shape，使用model的[Resize](https://www.mindspore.cn/lite/docs/zh-CN/master/mindir/runtime_cpp.html#%E5%8A%A8%E6%80%81shape%E8%BE%93%E5%85%A5)功能。

- 注意事项

    1）如果用户设置的动态维度数值过大或档位过多，可能会导致模型编译失败，此时建议用户减少档位或调低档位数值。<br/>
    2）如果用户设置了动态维度，实际推理时，使用的数据集图片大小需要与具体使用的维度相匹配。<br/>
    3）如果用户设置的动态维度数值过大或档位过多，在运行环境执行推理时，建议执行swapoff -a命令关闭swap交换区间作为内存的功能，防止出现由于内存不足，将swap交换空间作为内存继续调用，导致运行环境异常缓慢的情况。<br/>

## AOE自动调优

AOE是一款专门为Davinci平台打造的计算图形性能自动调优工具。Lite使能AOE的能力，是在converter阶段集成AOE离线可执行程序，对图进行性能调优，生成知识库，并保存离线模型。该功能支持子图调优和算子调优。具体使用流程如下：

### AOE工具调优

1. 配置环境变量

    ``${LOCAL_ASCEND}``为昇腾软件包安装所在路径

    ```bash
    export LOCAL_ASCEND=/usr/local/Ascend
    source ${LOCAL_ASCEND}/latest/bin/setenv.bash
    ```

    确认环境中AOE可执行程序可被找到并运行：

    ```bash
    aoe -h
    ```

2. 指定知识库路径

    AOE调优会生成算子知识库，默认的路径为

    ```bash
    ${HOME}/Ascend/latest/data/aoe/custom/graph(op)/${soc_version}
    ```

    （可选）也可通过``export TUNE_BANK_PATH``环境变量来自定义知识库路径。

3. 清除缓存

    为了模型编译能命中AOE生成的知识库，在使能AOE之前，最好先删除编译缓存，以免缓存复用，以Atlas推理系列产品环境，用户为root为例，删除``/root/atc_data/kernel_cache/Ascend310P3``和``/root/atc_data/fuzzy_kernel_cache/Ascend310P3``目录。

4. 配置文件指定选项

    在转换工具config配置文件中``[ascend_context]``指定AOE调优模式，如下举例中，会先执行子图调优，再执行算子调优。

    ```bash
    [ascend_context]
    aoe_mode="subgraph tuning, operator tuning"
    ```

> - 性能提升结果会因不同环境存在差异，实际时延减少百分比不完全等同于调优日志中所展示的结果。
> - AOE调优会在执行任务的当前目录下产生``aoe_workspace``目录，用于保存调优前后的模型，用于性能提升对比，以及调优所必须的过程数据和结果文件。该目录会占用额外磁盘空间，如500MB左右的原始模型会占用2~10GB的磁盘空间，视模型大小，算子种类结构，输入shape的大小等因素浮动。因此建议预留足够的磁盘空间，否则可能导致调优失败。
> - ``aoe_workspace``目录需要手动删除来释放磁盘空间。

### AOE API调优

Ascend推理时，运行时指定 `provider` 为 ``ge`` 时，支持多个模型共享权重，支持模型中存在可以被更新的权重，即变量。当前仅AOE API调优支持模型中存在变量，默认的AOE工具调优不支持。环境变量、知识库路径的设置和使用、AOE调优缓存与AOE工具调优一致。详情可参考[AOE调优](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/aoe/aoerc_16_0002.html)。

转换工具支持AOE API调优。当 `optimize=ascend_oriented`，配置文件中识别到 `[ascend_context]` 存在 `provider=ge` ，且 `[ascend_context]` 或 `[acl_option_cfg_param]` 中存在有效的 `aoe_mode` 或 `[aoe_global_options]` 存在有效的 `job_type` ，将启动AOE API调优。AOE API调优只产生知识库，不产生优化后的模型。

1. 指定 `provider` 为 ``ge``

    ```bash
    [ascend_context]
    provider=ge
    ```

2. AOE选项

    `[aoe_global_options]` 中的选项将传给AOE API的[全局选项](https://gitee.com/link?target=https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha003/developmenttools/devtool/aoe_16_070.html)。 `[aoe_tuning_options]` 中的选项将传给AOE API的[调优选项](https://gitee.com/link?target=https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha003/developmenttools/devtool/aoe_16_071.html)。

    我们将提取 `[acl_option_cfg_param]` 、`[ascend_context]` 、 `[ge_session_options]` 、 `[ge_graph_options]` 等Section中的选项并转换为AOE选项，避免用户开启AOE调优时需要手动转换这些选项。提取的选项包括 `input_format` 、 `input_shape` 、 `dynamic_dims` 、 `precision_mode` 。相同选项在多个配置Section同时存在时，优先级从前往后由低到高，`[aoe_global_options]` 和 `[aoe_tuning_options]` 中的选项优先级最高。建议使用 `[ge_graph_options]` 和 `[aoe_tuning_options]` 。

3. AOE调优模式

    `aoe_mode` 当前仅限定为 `subgraph tuning` 或 `operator tuning` ，暂不支持 `subgraph tuning, operator tuning`，即不支持同一个调优过程进行子图和算子调优，如需要，可通过两次调用转换工具分别启动子图调优和算子调优。

    `[aoe_global_options]` 中， `job_type` 为 ``1`` 时为子图调优， `job_type` 为 `2` 时为算子调优。

    ```bash
    [ascend_context]
    aoe_mode="operator tuning"
    ```

    ```bash
    [acl_option_cfg_param]
    aoe_mode="operator tuning"
    ```

    ```bash
    [aoe_global_options]
    job_type=2
    ```

4. 动态分档

    可在 `[acl_option_cfg_param]` 、`[ascend_context]` 、 `[ge_graph_options]` 、 `[aoe_tuning_options]` 设置动态分档信息，优先级从低到高。以下设置方式等价。 `[ascend_context]` 分档设置可参考 [动态shape配置](https://www.mindspore.cn/lite/docs/zh-CN/master/mindir/converter_tool_ascend.html#%E5%8A%A8%E6%80%81shape%E9%85%8D%E7%BD%AE)。 `[acl_option_cfg_param]` 、 `[ge_graph_options]` 、 `[aoe_tuning_options]` 分档设置可参考 [dynamic_dims](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/aoe/aoepar_16_013.html)、[dynamic_batch_size](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/aoe/aoepar_16_011.html)、[dynamic_image_size](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/aoe/aoepar_16_012.html)。注意， `[ge_graph_options]` 仅支持 `ge.dynamicDims` ，不支持类似 `dynamic_batch_size` 和 `dynamic_image_size` 的形式。 `input_format` 用于指定动态分档的输入维度排布，使用 `dynamic_image_size` 时需要指定 `input_format` 为 `NCHW` 或 `NHWC` 指示 `H` 和 `W` 维度所在位置。

    ```bash
    [ascend_context]
    input_shape=x1:[-1,3,224,224];x2:[-1,3,1024,1024]
    dynamic_dims=[1],[2],[3],[4];[1],[2],[3],[4]
    ```

    ```bash
    [acl_option_cfg_param]
    input_shape=x1:-1,3,224,224;x2:-1,3,1024,1024
    dynamic_dims=1,1;2,2;3,3;4,4
    ```

    ```bash
    [ge_graph_options]
    ge.inputShape=x1:-1,3,224,224;x2:-1,3,1024,1024
    ge.dynamicDims=1,1;2,2;3,3;4,4
    ```

    ```bash
    [aoe_tuning_options]
    input_shape=x1:-1,3,224,224;x2:-1,3,1024,1024
    dynamic_dims=1,1;2,2;3,3;4,4
    ```

5. 精度模式

    可在 `[acl_option_cfg_param]` 、`[ascend_context]` 、 `[ge_graph_options]` 、 `[aoe_tuning_options]` 设置模式信息，优先级从低到高。以下设置方式等价。

    `[ascend_context]` 和 `[acl_option_cfg_param]` 精度模式设置可参考 [ascend_context - precision_mode](https://www.mindspore.cn/lite/docs/zh-CN/master/mindir/converter_tool_ascend.html#%E9%85%8D%E7%BD%AE%E6%96%87%E4%BB%B6)。 `[ge_graph_options]` 和 `[aoe_tuning_options]` 精度模式设置可参考 [precision_mode](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devaids/devtools/aoe/aoepar_16_042.html)。

    ```bash
    [ascend_context]
    precision_mode=preferred_fp32
    ```

    ```bash
    [acl_option_cfg_param]
    precision_mode=preferred_fp32
    ```

    ```bash
    [ge_graph_options]
    precision_mode=allow_fp32_to_fp16
    ```

    ```bash
    [aoe_tuning_options]
    precision_mode=allow_fp32_to_fp16
    ```

## 部署Ascend自定义算子

MindSpore Lite converter支持将带有MindSpore Lite自定义Ascend算子的模型转换为MindSpore Lite的模型，通过自定义算子，可以在特殊场景下使用自定义算子对模型推理性能进行优化，如使用自定义的MatMul实现更高的矩阵乘法计算，使用MindSpore Lite提供的transformer融合算子提升transformer模型性能（待上线）以及使用AKG图算融合算子对模型进行自动融合优化提升推理性能等。

如果MindSpore Lite转换Ascend模型时有自定义算子，用户需要在调用converter之前部署自定义算子到ACL的算子库中才能正常完成转换，以下描述了部署Ascend自定义算子的关键步骤：

1. 配置环境变量

    ``${ASCEND_OPP_PATH}``为昇腾软件CANN包的算子库路径，通常是在昇腾软件安装路径下，默认一般是``/usr/local/Ascend/latest/opp``。

    ```bash
    export ASCEND_OPP_PATH=/usr/local/Ascend/latest/opp
    ```

2. 获取Ascend自定义算子包

    MindSpore Lite云侧推理包中会包含Ascend自定义算子包目录，其相对目录为``${LITE_PACKAGE_PATH}/tools/custom_kernels/ascend``，解压MindSpore Lite云侧推理包后，进入对应目录。

    ```bash
    tar zxf mindspore-lite-{version}-linux-{arch}.tar.gz
    cd tools/custom_kernels/ascend
    ```

3. 运行install.sh脚本部署自定义算子

    在算子包目录下运行安装脚本部署自定义算子。

    ```bash
    bash install.sh
    ```

4. 查看昇腾算子库目录检查是否安装成功

    完成部署自定义算子之后，进入昇腾算子库目录``/usr/local/Ascend/latest/opp/vendors/``，查看其下目录是否有对应的自定义算子文件，当前主要提供了基本算子样例和AKG图算融合算子实现，具体文件结构如下：

    ```text
    /usr/local/Ascend/latest/opp/vendors/
    ├── config.ini                                                     # 自定义算子vendor配置文件，定义不同vendor间优先级，需要有mslite的vendor配置
    └── mslite                                                         # mslite提供的自定义算子目录
        ├── framework                                                  # 第三方框架适配配置
        │    └── tensorflow                                            # tensorflow适配配置，非必需
        │       └── npu_supported_ops.json
        ├── op_impl                                                    # 自定义算子实现目录
        │   ├── ai_core                                                # 运行在ai_core的算子实现目录
        │   │   └── tbe                                                # tbe算子实现目录
        │   │       ├── config                                         # 不同芯片的算子配置
        │   │       │   ├── ascend310                                  # Atlas 200/300/500推理产品芯片的算子配置
        │   │       │       └── aic_ascend310-ops-info.json
        │   │       │   ├── ascend310p                                 # Atlas推理系列产品芯片的算子配置
        │   │       │       └── aic_ascend310p-ops-info.json
        │   │       │   ├── ascend910                                  # Atlas训练系列产品芯片的算子配置
        │   │       │       └── aic_ascend910-ops-info.json
        │   │       └── mslite_impl                                    # 算子的实现逻辑目录
        │   │           ├── add_dsl.py                                 # 基于dsl开发的add样例逻辑实现文件
        │   │           ├── add_tik.py                                 # 基于tik开发的add样例逻辑实现文件
        │   │           ├── compiler.py                                # akg图算需要的算子编译逻辑文件
        │   │           ├── custom.py                                  # akg自定义算子实现文件
        │   │           ├── matmul_tik.py                              # 基于tik开发的matmul样例逻辑实现文件
        │   ├── cpu                                                    # aicpu自定义算子目录，非必需
        │   │   └── aicpu_kernel
        │   │       └── impl
        │   └── vector_core                                            # 运行在vector_core的算子实现目录
        │       └── tbe                                                # tbe算子实现目录
        │           └── mslite_impl                                    # 算子的实现逻辑目录
        │               ├── add_dsl.py                                 # 基于dsl开发的add样例逻辑实现文件
        │               ├── add_tik.py                                 # 基于tik开发的add样例逻辑实现文件
        │               └── matmul_tik.py                              # 基于tik开发的matmul样例逻辑实现文件
        └── op_proto                                                   # 算子原型定义包目录
            └── libcust_op_proto.so                                    # 算子原型定义so文件，akg自定义算子默认注册，不需要此文件
    ```
