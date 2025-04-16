# Ascend Conversion Tool Description

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/docs/lite/docs/source_en/mindir/converter_tool_ascend.md)

## Introduction

This article introduces the related features of the cloud-side inference model conversion tool in Ascend back-end, such as profile options, dynamic shape, AOE, custom operators.

## Configuration File

Table 1: Configure ascend_context parameter

| Parameters  | Attributes  | Functions Description           | Types | Values Description |
| -------------------------- | ---- | ------------------------------------------------------------ | -------- | ------ |
| `input_format`             | Optional| Specify the model input format. | String | Options: `"NCHW"`, `"NHWC"`, and `"ND"` |
| `input_shape`       | Optional| Specify the model input Shape. input_name must be the input name in the network model before conversion, in the order of the inputs, separated by `;`. Only works for dynamic BatchSize. For static BatchSize, use converter_lite command to specify inputShape parameter. | String | Such as `"input1:[1,64,64,3];input2:[1,256,256,3]"` |
| `dynamic_dims`       | Optional| Specify the dynamic BatchSize and dynamic resolution parameters. | String | See [Dynamic shape configuration](#dynamic-shape-configuration) |
| `precision_mode`           | Optional| Configure the model accuracy mode.    | String | Options: `"enforce_fp32"`, `"preferred_fp32"`, `"enforce_fp16"`, `"enforce_origin"` or `"preferred_optimal"`. Default: `"enforce_fp16"`|
| `op_select_impl_mode`      | Optional| Configure the operator selection mode.    | String | Optioans: `"high_performance"`, and `"high_precision"`. Default: `"high_performance"`. |
| `output_type`       | Optional| Specify the data type of network output | String | Options: `"FP16"`, `"FP32"`, `"UINT8"` |
| `fusion_switch_config_file_path` | Optional| Configure the [Fusion Switch Configuration File](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/atctool/atctool_0078.html) file path and file name. | String   | Specify the configuration file for the fusion switch      |
| `insert_op_config_file_path` | Optional| Model insertion [AIPP](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/atctool/atctool_0018.html) operator | String  | Path of [AIPP](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/atctool/atctool_0021.html) configuration file |
| `aoe_mode` | Optional| [AOE](https://www.hiascend.com/document/detail/zh/TensorFlowCommunity/800alpha001/migration/tfmigr1/tfmigr1_000052.html) auto-tuning mode | String  | Options: "subgraph tuning", "operator tuning" or "subgraph tuning, operator tuning". Default: Not enabled |
| `plugin_custom_ops` | Optional | Enable Ascend backend fusion optimization to generate custom operators | String  | The available options are `All`, `None`, `FlashAttention`, `LayerNormV3`, `GeGluV2`, `GroupNormSilu`, `FFN`, `AddLayerNorm`, `MatMulAllReduce` and `BatchMatmulToMatmul`, where `All` means enabling `FlashAttention`, `LayerNormV3`, `GeGluV2` and `GroupNormSilu`, default `None` means not enabled |
| `custom_fusion_pattern` | Optional | Specify custom operator structures in the enabling model | String  | `custom operator type: original operator name in the model: enabled or disabled`, which can be taken as `enable` or `disable` |
| ` op-attrs ` | Optional | Specify custom operator attributes for fusion | String | `Custom operator name:Attribute:Value`. Currently, the operator only supports `FlashAttention`, which supports three optional configuration attributes: `input_layout`, `seq_threshold`, `inner_precise`, which respectively determine whether `FlashAttention` is fused in the form of `BNSD` (default), `BSH` or `BNSD_BSND`(`BNSD` means `FlashAttention` input and output `layout` is `BNSD`, `BSH` means input and output are `BSH`, `BNSD_BSND` means input is `BNSD`, output is `BSND`), the `seq` threshold (default `0`), and high-performance (default) or high-precision for `FlashAttention` fusion |
| `stream_label_file` | Optional | Specify the path of the operator multi-stream configuration file | String | `Multi-stream profile path`. The configuration file specifies which operators are executed in other streams in the format of `stream label: operator name 1, operator name 2`. The stream label is a string. There is already a stream in the model by default, and the operator configured in the configuration file will run the newly started stream. If you want to start more streams, you can configure multiple rows and configure a different stream label for each row. |

Table 2:  Configure [acl_init_options] parameter

| Parameters  | Attributes  | Functions Description           | Types | Values Description |
| -------------------------- | ---- | ------------------------------------------------------------ | -------- | ------ |
| `ge.engineType`              |  Optional  | Set the core type used by the network model. | String | Options: `"VectorCore"`, `"AiCore"` |
| `ge.socVersion`              |  Optional  | Version of the Ascend AI processor. | String | Options: `"Ascend310"`, `"Ascend710"`, `"Ascend910"` |
| `ge.bufferOptimize`          |  Optional  | Data cache optimization switch. | String | Options: `"l1_optimize"`, `"l2_optimize"`, `"off_optimize"`. Default: `"l2_optimize"` |
| `ge.enableCompressWeight`    |  Optional  | Data compression can be performed on Weight to improve performance. | String | Options: `"true"`, `"false"` |
| `compress_weight_conf`       |  Optional  | The path of the configuration file for the node list to be compressed is mainly composed of the conv operator and the fc operator. | String | path of config file |
| `ge.exec.precision_mode`     |  Optional  | Select the operator precision mode. | String | Options: `"force_fp32"`, `"force_fp16"`, `"allow_fp32_to_fp16"`, `"must_keep_origin_dtype"`, `"allow_mix_precision"`, Default: `"force_fp16"` |
| `ge.exec.disableReuseMemory` |  Optional  | Memory reuse switch. | String | Options: `"0"`, `"1"` |
| `ge.enableSingleStream`      |  Optional  | Whether to enable a model to use only one stream. | String | Options: `"true"`, `"false"` |
| `ge.aicoreNum`               |  Optional  | Set the number of AI cores used during compilation. | String | Default: `"10"` |
| `ge.fusionSwitchFile`        |  Optional  | Fusion configuration file path. | String | path of config file |
| `ge.enableSmallChannel`      |  Optional  | Whether to enable the optimization of small channel. | String | Options: `"0"`, `"1"` |
| `ge.opSelectImplmode`        |  Optional  | Select the operator implementation mode. | String | Options: `"high_precision"`, `"high_performance"` |
| `ge.optypelistForImplmode`   |  Optional  | A list of operators that uses the mode specified by the `ge.opSelectImplmode` parameter. | String | Operator type |
| `ge.op_compiler_cache_mode`  |  Optional  | Configure operator to compile Disk buffer mode. | String | Options: `"enable"`, `"force"`, `"disable"` |
| `ge.op_compiler_cache_dir`   |  Optional  | Configure operator mutation Disk buffer directory. | String | Options: `$HOME/atc_data` |
| `ge.debugDir`                |  Optional  | Configure the path to save debugging related process files generated by operator compilation. | String | Default Generate Current Path |
| `ge.opDebugLevel`            |  Optional  | Operator debug function switch. | String | Options: `"0"`, `"1"` |
| `ge.exec.modify_mixlist`     |  Optional  | Configure a mixed precision list. | String | path of config file |
| `ge.enableSparseMatrixWeight`|  Optional  | Enable global sparsity characteristics. | String | Options: `"1"`, `"0"` |
| `ge.externalWeight`          |  Optional  | Whether to save the weights of constant nodes separately in a file. | String | Options: `"1"`, `"0"` |
| `ge.deterministic`           |  Optional  | Whether to enable deterministic calculation. | String | Options: `"1"`, `"0"` |
| `ge.host_env_os`             |  Optional  | Support for inconsistency between the compilation environment operating system and the runtime environment. | String | Options: `"linux"` |
| `ge.host_env_cpu`            |  Optional  | Support for inconsistency between the operating system architecture and the runtime environment in the compilation environment. | String | Options: `"aarch64"`, `"x86_64"` |
| `ge.virtual_type`            |  Optional  | Whether the offline model is supported to run on virtual devices generated by the Ascend virtualization instance feature. | String | Options: `"0"`, `"1"` |
| `ge.compressionOptimizeConf` |  Optional  | Compression optimization feature configuration file path. | String | path of config file |

Table 3: Configure [acl_build_options] parameter

| Parameters  | Attributes  | Functions Description           | Types | Values Description |
| ----------------------------------- | ---- | ------------------------------------------------------------ | -------- | ------ |
| `input_format`                      | Optional | Specify the model input format. | String | Options: `"NCHW"`, `"NHWC"`, `"ND"` |
| `input_shape`                       | Optional | Specify the model input shape. After the model is converted, it can be obtained using the Model.get_model_info ("input_shape") method. This parameter is consistent with the command line input_shape. | String | For example: `input1:1,3,512,512;input2:1,3,224,224` |
| `input_shape_rang`                  | Optional | Specify the model shape rang. | String | For example: `input1:[1-10,3,512,512];input2:[1-10,3,224,224]` |
| `op_name_map`                       | Optional | Extension operator mapping configuration file path. | String | path of config file |
| `ge.dynamicBatchSize`               | Optional | Set dynamic batch gear parameters. | String | This parameter needs to be used in conjunction with the `input_shape` parameter |
| `ge.dynamicImageSize`               | Optional | Set dynamic resolution parameters for input images. | String | This parameter needs to be used in conjunction with the `input_shape` parameter |
| `ge.dynamicDims`                    | Optional | Set the gear of dynamic dimensions in ND format. After the model is converted, it can be obtained using the Model.get_model_info ("dynamic_dims") method | String | This parameter needs to be used in conjunction with the `input_shape` parameter |
| `ge.inserOpFile`                    | Optional | Enter the configuration file path for the preprocessing operator. | String | path of config file |
| `ge.exec.precision_mode`            | Optional | Enter the configuration file path for the preprocessing operator. | String | Options: `"force_fp32"`, `"force_fp16"`, `"allow_fp32_to_fp16"`, `"must_keep_origin_dtype"`, `"allow_mix_precision"`, Default: `"force_fp16"` |
| `ge.exec.disableReuseMemory`        | Optional | Memory reuse switch. | String | Options: `"0"`, `"1"` |
| `ge.outputDataType`                 | Optional | Network output data type. | String | Options: `"FP32"`, `"UINT8"`, `"FP16"` |
| `ge.outputNodeName`                 | Optional | Specify output nodes. | String | For example: `"node_name1:0;node_name1:1;node_name2:0"` |
| `ge.INPUT_NODES_SET_FP16`           | Optional | Specify the input node name with input data type FP16. | String | `"node_name1;node_name2"` |
| `log`                               | Optional | Set Log Level. | String | Options: `"debug"`, `"info"`, `"warning"`, `"error"` |
| `ge.op_compiler_cache_mode`         | Optional | Configure operator to compile disk buffer mode. | String | Options: `"enable"`, `"force"`, `"disable"` |
| `ge.op_compiler_cache_dir`          | Optional | Configure operator mutation disk buffer directory. | String | Options: `$HOME/atc_data` |
| `ge.debugDir`                       | Optional | Configure the path to save debugging related process files generated by operator compilation. | String | Default Generate Current Path |
| `ge.opDebugLevel`                   | Optional | Operator debug function switch. | String | Options: `"0"`, `"1"` |
| `ge.mdl_bank_path`                  | Optional | Path to custom knowledge base after loading model tuning. | String | This parameter needs to be used in conjunction with the `ge.bufferOptimize` parameter |
| `ge.op_bank_path`                   | Optional | Customizing the knowledge base path after loading operator tuning. | String | Knowledge Base Path |
| `ge.exec.modify_mixlist`            | Optional | Configure mixed precision list. | String | path of config file |
| `ge.exec.op_precision_mode`         | Optional | Set the precision mode of a specific operator and use this parameter to set the configuration file path. | String | path of config file |
| `ge.shape_generalized_build_mode`   | Optional | Shape compilation method during image compilation. | String | Options: `"shape_generalized"`, `"shape_precise"` |
| `op_debug_config`                   | Optional | Memory detection function switch. | String | path of config file |
| `ge.externalWeight`                 | Optional | Do you want to save the weights of constant nodes separately in a file. | String | Options: `"1"`, `"0"` |
| `ge.exec.exclude_engines`           | Optional | Set the network model not to use one or some acceleration engines. | String | Options: `"AiCore"`, `"AiVec"`, `"AiCpu"` |

## Dynamic Shape Configuration

In some inference scenarios, such as detecting a target and then executing the target recognition network, the number of targets is not fixed resulting in a variable input BatchSize for the target recognition network. If each inference is computed at the maximum BatchSize or maximum resolution, it will result in wasted computational resources. Therefore, it needs to support dynamic BatchSize and dynamic resolution scenarios during inference. Lite inference on Ascend supports dynamic BatchSize and dynamic resolution scenarios. The dynamic_dims dynamic parameter in [ascend_context] is configured via configFile in the convert phase, and the model [Resize](https://www.mindspore.cn/lite/docs/en/r2.6.0/mindir/runtime_cpp.html#dynamic-shape-input) is used during inference, to change the input shape.

### Dynamic Batch Size

- Parameter Name

    dynamic_dims

- Functions

    Set the dynamic batch profile parameter for scenarios where the number of images processed at a time is not fixed during inference. This parameter needs to be used in conjunction with input_shape, and the position of -1 in input_shape is the dimension where the dynamic batch is located.

- Value

    Support up to 100 profiles configuration. Each profile is separated by English comma. The value limit of each profile: [1~2048]. For example, the parameters in the configuration file are configured as follows:

    ```
    [ascend_context]
    input_shape=input:[-1,64,64,3]
    dynamic_dims=[1],[2]
    ```

    "-1" in input_shape means setting dynamic batch, and the profile can take the value of "1,2", that is, support profile 0: [1,64,64,3], profile 1: [2,64,64,3].

    If more than one input exists, the profiles corresponding to the different inputs needs to be the same and separated by `;`.

    ```
    [ascend_context]
    input_shape=input1:[-1,64,64,3];input2:[-1,256,256,3]
    dynamic_dims=[1],[2];[1],[2]
    ```

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --configFile=./config.txt --optimize=ascend_oriented --outputFile=${model_name}
    ```

    Note: When enabling dynamic BatchSize, you do not need to specify the inputShape parameter, and only need to configure the [ascend_context] dynamic batch size through configFile, that is, the configuration content in the previous section.

- Inference

    Enable dynamic BatchSize. When the model inference is performed, the input shape can only choose the set value of the profile at the time of the converter. If you want to switch to the input shape corresponding to another profile, use the model [Resize](https://www.mindspore.cn/lite/docs/en/r2.6.0/mindir/runtime_cpp.html#dynamic-shape-input) function.

- Precautions

    1. If the user performs inference operations and the number of images processed is not fixed at a time, this parameter can be configured to dynamically allocate the number of images processed at a time. For example, if a user needs to process 2, 4, or 8 images each time to perform inference, it can be configured as 2, 4, and 8. Once the profile is requested, memory will be requested based on the actual profile during model inference.<br/>
    2. If the profile value set by the user is too large or the profiles are too many, it may cause model compilation failure, in which case the user is advised to reduce the profiles or turn down the profile value.<br/>
    3. If the profile value set by the user is too large or the profiles are too many, when performing inference in the runtime environment, it is recommended that the swapoff -a command be executed to turn off the swap interval as memory, to avoid that the swap space is continued to be called as memory, resulting in an unusually slow running environment due to the lack of memory.<br/>

### Dynamic Resolution

- Parameter Name

    dynamic_dims

- Function

    Set the dynamic resolution parameter of the input image. For scenarios where the width and height of the image are not fixed each time during inference. This parameter needs to be used in conjunction with input_shape, and the position of -1 in input_shape is the dimension where the dynamic resolution is located.

- Value

    Support up to 100 profiles configuration. Each profile is separated by English comma, such as "[imagesize1_height,imagesize1_width],[imagesize2_height,imagesize2_width]". For example, the parameters in the configuration file are configured as follows:

    ```
    [ascend_context]
    input_format=NHWC
    input_shape=input:[1,-1,-1,3]
    dynamic_dims=[64,64],[19200,960]
    ```

    "-1" in input_shape means setting the dynamic resolution, i.e., it supports profile 0: [1,64,64,3] and profile 1: [1,19200,960,3].

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --configFile=./config.txt --optimize=ascend_oriented --outputFile=${model_name}
    ```

    Note: When enabling dynamic BatchSize, you do not need to specify the inputShape parameter, and only need to configure the [ascend_context] dynamic resolution through configFile, that is, the configuration content in the previous section.

- Inference

    By enabling dynamic resolution, when model inference is performed, the input shape can only select the set profile value at the time of the converter. If you want to switch to the input shape corresponding to another profile, use the model [Resize](https://www.mindspore.cn/lite/docs/en/r2.6.0/mindir/runtime_cpp.html#dynamic-shape-input) function.

- Precautions

    1. If the resolution value set by the user is too large or the profiles are too many, it may cause model compilation failure, in which case the user is advised to reduce the profiles or turn down the profile value.<br/>
    2. If the user sets a dynamic resolution, the size of the dataset images used for the actual inference needs to match the specific resolution used.<br/>
    3. If the resolution value set by the user is too large or the profiles are too many, when performing inference in the runtime environment, it is recommended that the swapoff -a command be executed to turn off the swap interval as memory, to avoid that the swap space is continued to be called as memory, resulting in an unusually slow running environment due to the lack of memory.<br/>

### Dynamic dimension

- Parameter Name

    `ge.dynamicDims`

- Function

    Set the gear of the dynamic dimension input in ND format. Applicable to scenarios where any dimension is processed each time reasoning is performed, This parameter needs to be used in conjunction with input_shape, and the position of -1 in input_shape is the dimension where the dynamic dim is located.

- Value

    Up to 100 configurations are supported, each separated by an English comma. For example, the parameters in the configuration file are configured as follows:

    ```
    [acl_build_options]
    input_format="ND"
    input_shape="input1:1,-1,-1;input2:1,-1"
    ge.dynamicDims="32,32,24;64,64,36"
    ```

    The "-1" in the shape indicates the setting of dynamic dimensions, which supports gear 0: input1:1,32,32; input2:1,24, gear 1:1, 64,64; input2:1,36.

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --configFile=./config.txt --optimize=ascend_oriented --outputFile=${model_name}
    ```

    Note: When enabling dynamic dimension, `input_format` must be set to `ND`.

- Inference

    By enabling dynamic dimension, when model inference is performed, the input shape can only select the set profile value at the time of the converter. If you want to switch to the input shape corresponding to another profile, use the model [Resize](https://www.mindspore.cn/lite/docs/en/r2.6.0/mindir/runtime_cpp.html#dynamic-shape-input) function.

- Precautions

    1. If the resolution value set by the user is too large or the profiles are too many, it may cause model compilation failure, in which case the user is advised to reduce the profiles or turn down the profile value.<br/>
    2. If the user sets a dynamic dimension, the dimension of the inputs for the actual inference needs to match the specific resolution used.<br/>
    3. If the resolution value set by the user is too large or the profiles are too many, when performing inference in the runtime environment, it is recommended that the swapoff -a command be executed to turn off the swap interval as memory, to avoid that the swap space is continued to be called as memory, resulting in an unusually slow running environment due to the lack of memory.<br/>

## AOE Auto-tuning

AOE is a computational graph performance auto-tuning tool built specifically for the Davinci platform. Lite enables AOE ability to integrate the AOE offline executable in the converter phase, to perform performance tuning of the graph, generate a knowledge base, and save the offline model. This function supports subgraph tuning and operator tuning. The function supports subgraph tuning and operator tuning. The specific use process is as follows:

### AOE Tool Tuning

1. Configure environment variables

    ``${LOCAL_ASCEND}`` is the path where the Ascend package is installed

    ```bash
    export LOCAL_ASCEND=/usr/local/Ascend
    source ${LOCAL_ASCEND}/latest/bin/setenv.bash
    ```

    Confirm that the AOE executable program can be found and run in the environment:

    ```bash
    aoe -h
    ```

2. Specify the knowledge base path

    AOE tuning generates an operator knowledge base. The default path:

    ```bash
    ${HOME}/Ascend/latest/data/aoe/custom/graph(op)/${soc_version}
    ```

    (Optional) You can also customize the knowledge base path with the ``export TUNE_BANK_PATH`` environment variable.

3. Clear the cache

    In order for the model compilation to get the knowledge base generated by AOE, it is best to delete the compilation cache before AOE is enabled to avoid cache reuse. Taking Atlas inference series environment with user as root for example, delete ``/root/atc_data/kernel_cache/Ascend310P3`` and ``/root/atc_data/fuzzy_kernel_cache/Ascend310P3`` directories.

4. Specified options of the configuration file

    Specify the AOE tuning mode in the ``[ascend_context]`` configuration file of the conversion tool config. In the following example, the subgraph tuning will be executed first, and then the operator tuning.

    ```bash
    [ascend_context]
    aoe_mode="subgraph tuning, operator tuning"
    ```

> - The performance improvements will vary from environment to environment, and the actual latency reduction percentage is not exactly the same as the results shown in the tuning logs.
> - AOE tuning generates ``aoe_workspace`` directory in the current directory where the task is executed, which is used to save the models before and after tuning for performance improvement comparison, as well as the process data and result files necessary for tuning. This directory will occupy additional disk space, e.g., 2~10GB for a 500MB raw model, depending on the model size, operator type structure, input shape size and other factors. Therefore, it is recommended to reserve enough disk space, otherwise it may lead to tuning failure.
> - The ``aoe_workspace`` directory needs to be deleted manually to free up disk space.

### AOE API Tuning

For Ascend inference, when the runtime specifies `provider` as ``ge``, multiple models within one device can share weights, and some the weights in the model can be updated, that is, variables. Currently, only AOE API tuning supports variables exists in the model, and the default AOE tool tuning does not support that. The environment variables, setting and use of knowledge base paths, and AOE tuning cache are consistent with AOE tool tuning in the previous section. For details, please refer to [AOE tuning](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/aoe/aoerc_16_0002.html).

AOE API tuning needs to be done through converter tool. When `optimize=ascend_oriented`, in the configuration file, there is `provider=ge` in `[ascend_context]`, and there is a valid `aoe_mode` in `[ascend_context]` or `acl_option_cfg_param]`, or there is a valid `job_type` in `[aoe_global_options]`, AOE API tuning will be performed. AOE API tuning only generates a knowledge base and does not generate an optimized model.

1. Specify `provider` as ``ge``

    ```bash
    [ascend_context]
    provider=ge
    ```

2. AOE options

    The options in `[aoe_global_options]` will be passed through to the [global options](https://gitee.com/link?target=https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha003/developmenttools/devtool/aoe_16_070.html) of the AOE API. The options in `[aoe_tuning_options]` will be passed through to the [tuning options](https://gitee.com/link?target=https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/63RC2alpha003/developmenttools/devtool/aoe_16_071.html) of the AOE API.

    We will extract the options in sections `[acl_option_cfg_param]`, `[ascend_context]`, `[ge_session_options]` and `[ge_graph_options]` and convert them into AOE options to avoid the need for users to manually convert these options. The extracted options include `input_format`, `input_shape`, `dynamic_dims` and `precision_mode`. When the same option exists in multiple configuration sections at the same time, the priority ranges from low to high, with options in `[aoe_global_options]` and `[aoe_tuning_options]` having the highest priority. It is recommended to use `[ge_graph_options]` and `aoe_uning_options`.

3. AOE tuning mode

    The `aoe_mode` is currently limited to `subgraph tuning` or `operator tuning`. Currently, `subgraph tuning, operator tuning` is not supported, which means that subgraph and operator tuning is not supported in the same tuning process. If necessary, subgraph and operator tuning can be performed separately.

    In `[aoe_global_options]`, when the value of `job_type` is ``1``, it means subgraph tuning, and when the value is ``2``, it means operator tuning.

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

4. Dynamic dimension profiles

    Dynamic dimension profiles can be set in `[acl_option_cfg_param]`, `[ascend_context]`, `[ge_graph_options]`, `[aoe_tuning_options]`, with priority ranging from low to high. The following settings are equivalent. Setting the dynamic dimension profiles in `[ascend_context]` can refer to [Dynamic Shape Configuration](https://www.mindspore.cn/lite/docs/en/r2.6.0/mindir/converter_tool_ascend.html#dynamic-shape-configuration). Setting the dynamic dimension profiles in `[acl_option_cfg_param]`, `[ge_graph_options]` and `[aoe_tuning_options]` can refer to [dynamic_dims](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/aoe/aoepar_16_013.html), [dynamic_batch_size](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/aoe/aoepar_16_011.html), [dynamic_image_size](https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/devaids/devtools/aoe/aoepar_16_012.html). Note that the `[ge_graph_options]` only supports the `ge.dynamicDims` and does not support the forms of `dynamic_batch_size` and `dynamic_image_size`. `input_format` is used to specify the input dimension layout for dynamic profiles. When using `dynamic_image_size`, it is necessary to specify `input_format` as `NCHW` or `NHWC` to indicate the location of the `H` and `W` dimensions.

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

5. Precision mode

    Precision mode can be set in `[acl_option_cfg_param]`, `[ascend_context]`, `[ge_graph_options]`, `[aoe_tuning_options]`, with priority ranging from low to high. The following settings are equivalent. Setting the precision mode in `[ascend_context]` and `[acl_option_cfg_param]` can refer to [ascend_context - precision_mode](https://www.mindspore.cn/lite/docs/en/r2.6.0/mindir/converter_tool_ascend.html#configuration-file). Setting the precision mode in `[ge_graph_options]` and `[aoe_tuning_options]` can refer to [precision_mode](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha001/devaids/devtools/aoe/aoepar_16_042.html).

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

## Deploying Ascend Custom Operators

MindSpore Lite converter supports converting models with MindSpore Lite custom Ascend operators to MindSpore Lite models. Custom operators can be used to optimize model inference performance in special scenarios, such as using custom MatMul to achieve higher matrix multiplication, using the transformer fusion operators provided by MindSpore Lite to improve transformer model performance (to be launched) and using the AKG graph fusion operator to automatically fuse models to improve inference performance.

If MindSpore Lite converts Ascend models with custom operators, user needs to deploy the custom operators to the ACL operator library before calling the converter in order to complete the conversion properly. The following describes the key steps to deploy Ascend custom operators:

1. Configure environment variables

    ``${ASCEND_OPP_PATH}`` is the operator library path of Ascend software CANN package, usually under Ascend software installation path. The default is usually ``/usr/local/Ascend/latest/opp``.

    ```bash
    export ASCEND_OPP_PATH=/usr/local/Ascend/latest/opp
    ```

2. Obtain Ascend custom operator package

    MindSpore Lite cloud-side inference package will contain Ascend custom operator package directory whose relative directory is ``${LITE_PACKAGE_PATH}/tools/custom_kernels/ascend``. After unzip the MindSpore Lite cloud-side inference package, enter the corresponding directory.

    ```bash
    tar zxf mindspore-lite-{version}-linux-{arch}.tar.gz
    cd tools/custom_kernels/ascend
    ```

3. Run install.sh script to deploy custom operator

    Run the installation script in the operator package directory to deploy the custom operator.

    ```bash
    bash install.sh
    ```

4. Check the Ascend library directory to see if the installation is successful

    After deploying the custom operator, go to the Ascend operator library directory ``/usr/local/Ascend/latest/opp/vendors/`` and check whether there are corresponding custom operator files in the directory. At present, we mainly provide the basic operator sample and the AKG graph fusion operator implementation. The specific file structure is as follows:

    ```text
    /usr/local/Ascend/latest/opp/vendors/
    ├── config.ini                                                     # Custom operator vendor configuration file, define the priority between different vendors, which needs to have vendor configuration of mslite
    └── mslite                                                         # Custom operator directory provided by mslite
        ├── framework                                                  # Third-party framework adaptation configuration
        │    └── tensorflow                                            # tensorflow adaptation configuration, not required
        │       └── npu_supported_ops.json
        ├── op_impl                                                    # Custom operator implementation directory
        │   ├── ai_core                                                # Run operator implementation directory in ai_core
        │   │   └── tbe                                                # tbe operator implementation directory
        │   │       ├── config                                         # Operator configurations for different chips
        │   │       │   ├── ascend310                                  # Operator configuration of Atlas 200/300/500 inference product chip
        │   │       │       └── aic_ascend310-ops-info.json
        │   │       │   ├── ascend310p                                 # Operator configuration of Atlas inference series chip
        │   │       │       └── aic_ascend310p-ops-info.json
        │   │       │   ├── ascend910                                  # Operator configuration of Atlas training series chip
        │   │       │       └── aic_ascend910-ops-info.json
        │   │       └── mslite_impl                                    # Implementation logic directory of operators
        │   │           ├── add_dsl.py                                 # add sample logic implementation file based on dsl development
        │   │           ├── add_tik.py                                 # add sample logic implementation file based on tik development
        │   │           ├── compiler.py                                # Operator compilation logic file needed for akg graph
        │   │           ├── custom.py                                  # akg custom operator implementation file
        │   │           ├── matmul_tik.py                              # matmul sample logic implementation file based on tik development
        │   ├── cpu                                                    # aicpu custom operator subdirectory, not required
        │   │   └── aicpu_kernel
        │   │       └── impl
        │   └── vector_core                                            # Run operator implementation directory in vector_core
        │       └── tbe                                                # tbe operator implementation directory
        │           └── mslite_impl                                    # Implementation logic directory of operators
        │               ├── add_dsl.py                                 # add sample logic implementation file based on dsl development
        │               ├── add_tik.py                                 # add sample logic implementation file based on tik development
        │               └── matmul_tik.py                              # matmul sample logic implementation file based on tik development
        └── op_proto                                                   # Operator prototype definition package directory
            └── libcust_op_proto.so                                    # operator prototype definition so file. akg custom operator is registered by default, and do not need this file
    ```