# Ascend Conversion Tool Description

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/lite/docs/source_en/use/cloud_infer/converter_tool_ascend.md)

## Introduction

This article introduces the related features of the cloud-side inference model conversion tool in Ascend back-end, such as profile options, dynamic shape, AOE, custom operators.

## Configuration File

Table 1: Configure [ascend_context] parameter

| Parameters  | Attributes  | Functions Description           | Types | Values Description |
| -------------------------- | ---- | ------------------------------------------------------------ | -------- | ------ |
| `input_format`             | Optional| Specify the model input format. | String | Options: `"NCHW"`, `"NHWC"`, and `"ND"` |
| `input_shape`       | Optional| Specify the model input Shape. input_name must be the input name in the network model before conversion, in the order of the inputs, separated by `;`. | String | Such as `"input1:[1,64,64,3];input2:[1,256,256,3]"` |
| `dynamic_dims`       | Optional| Specify the dynamic BatchSize and dynamic resolution parameters. | String | 见[Dynamic shape configuration](#dynamic-shape-configuration) |
| `precision_mode`           | Optional| Configure the model accuracy mode.    | String | Options: `"enforce_fp32"`, `"preferred_fp32"`, `"enforce_fp16"`, `"enforce_origin"` or `"preferred_optimal"`. Default: `"enforce_fp16"`|
| `op_select_impl_mode`      | Optional| Configure the operator selection mode.    | String | Optioans: `"high_performance"`, and `"high_precision"`. Default: `"high_performance"`. |
| `output_type`       | Optional| Specify the data type of network output | String | Options: `"FP16"`, `"FP32"`, `"UINT8"` |
| `fusion_switch_config_file_path` | Optional| Configure the [Fusion Switch Configuration File](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/atctool/atctool_0078.html) file path and file name. | String   | Specify the configuration file for the fusion switch      |
| `insert_op_config_file_path` | Optional| Model insertion [AIPP](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/atctool/atctool_0018.html) operator | String  | Path of [AIPP](https://www.hiascend.com/document/detail/en/canncommercial/601/inferapplicationdev/atctool/atctool_0021.html) configuration file |
| `aoe_mode` | Optional| [AOE](https://www.hiascend.com/document/detail/en/canncommercial/601/devtools/auxiliarydevtool/aoe_16_001.html) auto-tuning mode | String  | Options: "subgraph turing", "operator turing" or "subgraph turing, operator turing". Default: Not enabled |

## Dynamic Shape Configuration

In some inference scenarios, such as detecting a target and then executing the target recognition network, the number of targets is not fixed resulting in a variable input BatchSize for the target recognition network. If each inference is computed at the maximum BatchSize or maximum resolution, it will result in wasted computational resources. Therefore, it needs to support dynamic BatchSize and dynamic resolution scenarios during inference. Lite inference on Ascend supports dynamic BatchSize and dynamic resolution scenarios. The dynamic_dims dynamic parameter in [ascend_context] is configured via congFile in the convert phase, and the model [Resize](https://www.mindspore.cn/lite/docs/en/r2.0/use/cloud_infer/runtime_cpp.html#dynamic-shape-input) is used during inference, to change the input shape.

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

    Enable dynamic BatchSize. When the model inference is performed, the input shape can only choose the set value of the profile at the time of the converter. If you want to switch to the input shape corresponding to another profile, use the model [Resize](https://www.mindspore.cn/lite/docs/en/r2.0/use/cloud_infer/runtime_cpp.html#dynamic-shape-input) function.

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

    By enabling dynamic resolution, when model inference is performed, the input shape can only select the set profile value at the time of the converter. If you want to switch to the input shape corresponding to another profile, use the model [Resize](https://www.mindspore.cn/lite/docs/en/r2.0/use/cloud_infer/runtime_cpp.html#dynamic-shape-input) function.

- Precautions

    1. If the resolution value set by the user is too large or the profiles are too many, it may cause model compilation failure, in which case the user is advised to reduce the profiles or turn down the profile value.<br/>
    2. If the user sets a dynamic resolution, the size of the dataset images used for the actual inference needs to match the specific resolution used.<br/>
    3. If the resolution value set by the user is too large or the profiles are too many, when performing inference in the runtime environment, it is recommended that the swapoff -a command be executed to turn off the swap interval as memory, to avoid that the swap space is continued to be called as memory, resulting in an unusually slow running environment due to the lack of memory.<br/>

## AOE Auto-tuning

AOE is a computational graph performance auto-tuning tool built specifically for the Davinci platform. Lite enables AOE ability to integrate the AOE offline executable in the converter phase, to perform performance tuning of the graph, generate a knowledge base, and save the offline model. This function supports subgraph tuning and operator tuning. The function supports subgraph tuning and operator tuning. The specific use process is as follows:

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

    In order for the model compilation to get the knowledge base generated by AOE, it is best to delete the compilation cache before AOE is enabled to avoid cache reuse. Taking Ascend 310P environment with user as root for example, delete ``/root/atc_data/kernel_cache/Ascend310P3`` and ``/root/atc_data/fuzzy_kernel_cache/Ascend310P3`` directories.

4. Specified options of the configuration file

    Specify the AOE tuning mode in the ``[ascend_context]`` configuration file of the conversion tool config. In the following example, the subgraph tuning will be executed first, and then the operator tuning.

    ```bash
    [ascend_context]
    aoe_mode="subgraph tuning, operator tuning"
    ```

> - The performance improvements will vary from environment to environment, and the actual latency reduction percentage is not exactly the same as the results shown in the tuning logs.
> - AOE tuning generates ``aoe_workspace`` directory in the current directory where the task is executed, which is used to save the models before and after tuning for performance improvement comparison, as well as the process data and result files necessary for tuning. This directory will occupy additional disk space, e.g., 2~10GB for a 500MB raw model, depending on the model size, operator type structure, input shape size and other factors. Therefore, it is recommended to reserve enough disk space, otherwise it may lead to tuning failure.
> - The ``aoe_workspace`` directory needs to be deleted manually to free up disk space.