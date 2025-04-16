# Integrated Ascend

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0/docs/lite/docs/source_en/advanced/third_party/ascend_info.md)

> - The Ascend backend support on device-side version will be deprecated later. For related usage of the Ascend backend, please refer to the cloud-side inference version documentation.
> - [Build Cloud-side MindSpore Lite](https://mindspore.cn/lite/docs/en/r2.6.0/mindir/build.html)
> - [Cloud-side Model Converter](https://mindspore.cn/lite/docs/en/r2.6.0/mindir/converter.html)
> - [Cloud-side Benchmark Tool](https://mindspore.cn/lite/docs/en/r2.6.0/mindir/benchmark.html)

This document describes how to use MindSpore Lite to perform inference and use the dynamic shape function on Linux in the Ascend environment. Currently, MindSpore Lite supports the Atlas 200/300/500 inference product and Atlas inference series.

## Environment Preparation

### Checking System Environment Information

- Ensure that a 64-bit OS is installed, the [glibc](https://www.gnu.org/software/libc/) version is 2.17 or later, and Ubuntu 18.04, CentOS 7.6, and EulerOS 2.8 are verified.

- Ensure that [GCC 7.3.0](https://gcc.gnu.org/releases.html) is installed.

- Ensure that [CMake 3.18.3 or later](https://cmake.org/download/) is installed.
    - Ensure that the path of the installed CMake is added to a system environment variable.

- Ensure that Python 3.7.5 or 3.9.0 is installed. If neither of them is installed, you can download and install them via the following links:

    - Links for Python 3.7.5 (64-bit): [Official Website](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.7.5/Python-3.7.5.tgz)
    - Links for Python 3.9.0 (64-bit): [Official Website](https://www.python.org/ftp/python/3.9.0/Python-3.9.0.tgz) or [HUAWEI CLOUD](https://mirrors.huaweicloud.com/python/3.9.0/Python-3.9.0.tgz)

- If you use the ARM architecture, ensure that the pip version matching Python is 19.3 or later.

- Ensure that Ascend AI processor software package is installed.

    - Ascend software package provides two distributions, commercial edition and community edition:

        1. Commercial edition needs approval from Ascend to download, for detailed installation guide, please refer to [Ascend Data Center Solution 22.0.RC3 Installation Guide](https://support.huawei.com/enterprise/zh/doc/EDOC1100280094).

        2. Community edition has no restrictions, choose `5.1.RC2.alpha007` in [CANN community edition](https://www.hiascend.com/software/cann/community-history), then choose relevant driver and firmware packages in [firmware and driver](https://www.hiascend.com/hardware/firmware-drivers?tag=community). Please refer to the abovementioned commercial edition installation guide to choose which packages are to be installed and how to install them.

    - The default installation path of the installation package is `/usr/local/Ascend`. Ensure that the current user has the right to access the installation path of Ascend AI processor software package. If not, the root user needs to add the current user to the user group where `/usr/local/Ascend` is located.
    - Install the .whl packages provided in Ascend AI processor software package. If the .whl packages have been installed before, you should uninstall the .whl packages by running the following command.

        ```bash
        pip uninstall te topi -y
        ```

        Run the following command to install the .whl packages if the Ascend AI package has been installed in default path. If the installation path is not the default path, you need to replace the path in the command with the installation path.

        ```bash
        pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/topi-{version}-py3-none-any.whl
        pip install /usr/local/Ascend/ascend-toolkit/latest/lib64/te-{version}-py3-none-any.whl
        ```

### Configuring Environment Variables

After the Ascend software package is installed, export runtime environment variables. In the following command, `/usr/local/Ascend` in `LOCAL_ASCEND=/usr/local/Ascend` indicates the installation path of the software package. Change it to the actual installation path.

```bash
# control log level. 0-EBUG, 1-INFO, 2-WARNING, 3-ERROR, 4-CRITICAL, default level is WARNING.
export GLOG_v=2

# Conda environmental options
LOCAL_ASCEND=/usr/local/Ascend # the root directory of run package

# lib libraries that the run package depends on
export LD_LIBRARY_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/lib64:${LOCAL_ASCEND}/driver/lib64:${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}

# Environment variables that must be configured
export TBE_IMPL_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp/op_impl/built-in/ai_core/tbe            # TBE operator implementation tool path
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/opp                                       # OPP path
export PATH=${LOCAL_ASCEND}/ascend-toolkit/latest/compiler/ccec_compiler/bin/:${PATH}                  # TBE operator compilation tool path
export PYTHONPATH=${TBE_IMPL_PATH}:${PYTHONPATH}                                                       # Python library that TBE implementation depends on
```

## Executing the Converter

MindSpore Lite provides an offline model converter to convert various models (Caffe, ONNX, TensorFlow, and MindIR) into models that can be inferred on the Ascend hardware.
First, use the converter to convert a model into an `ms` model. Then, use the runtime inference framework matching the converter to perform inference. The process is as follows:

1. [Download](https://www.mindspore.cn/lite/docs/en/r2.6.0/use/downloads.html) the converter dedicated for Ascend. Currently, only Linux is supported.

2. Decompress the downloaded package.

     ```bash
     tar -zxvf mindspore-lite-{version}-linux-x64.tar.gz
     ```

   {version} indicates the version number of the release package.

3. Add the dynamic link library required by the converter to the environment variable LD_LIBRARY_PATH.

    ```bash
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PACKAGE_ROOT_PATH}/tools/converter/lib
    ```

   ${PACKAGE_ROOT_PATH} indicates the path of the folder obtained after the decompression.

4. Go to the converter directory.

    ```bash
    cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
    ```

5. (Optional) Configuring configFile

    You can use this option to configure the Ascend option for model conversion. The configuration file is in the INI format. For the Ascend scenario, the configurable parameter is [acl_option_cfg_param]. For details about the parameter, see the following table,  Ascend initialization can be configured through the acl_init_options parameter, and Ascend composition can be configured through the acl_build_options parameter.

6. Execute the converter to generate an Ascend `ms` model.

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --outputFile=${model_name}
    ```

    ${model_name} indicates the model file name. The execution result is as follows:

    ```text
    CONVERT RESULT SUCCESS:0
    ```

    For details about parameters of the converter_lite converter, see ["Parameter Description" in Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/r2.6.0/converter/converter_tool.html#parameter-description).

    Note: If the input shape of the original model is uncertain, specify inputShape when using the converter to convert a model. In addition, set configFile to the value of input_shape_vector parameter in acl_option_cfg_param. The command is as follows:

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --outputFile=${model_name} --inputShape="input:1,64,64,1" --configFile="./config.txt"
    ```

    The content of the config.txt file is as follows:

    ```cpp
    [acl_option_cfg_param]
    input_shape_vector="[1,64,64,1]"
    ```

Table 1 [acl_option_cfg_param] parameter configuration

| Parameter| Attribute| Function| Type| Value|
| -------------------------- | ---- | ------------------------------------------------------------ | -------- | ------ |
| `input_format`             | Optional| Specifies the model input format.| String | `"NCHW"` or `"NHWC"`|
| `input_shape_vector`       | Optional| Specifies the model input shapes which are arranged based on the model input sequence and are separated by semicolons (;).| String | Example: `"[1,2,3,4];[4,3,2,1]"`|
| `precision_mode`           | Optional| Configures the model precision mode.| String | `"force_fp16"` (default value), `"allow_fp32_to_fp16"`, `"must_keep_origin_dtype"`, or `"allow_mix_precision"`|
| `op_select_impl_mode`      | Optional| Configures the operator selection mode.| String | `"high_performance"` (default value) or `"high_precision"`|
| `dynamic_batch_size`       | Optional| Specifies the [dynamic batch size](#dynamic-batch-size) parameter.| String | `"2,4"`|
| `dynamic_image_size`       | Optional| Specifies the [dynamic image size](#dynamic-image-size) parameter.| String | `"96,96;32,32"` |
| `fusion_switch_config_file_path` | Optional| Configure the path and name of the [fusion pattern switch](https://www.hiascend.com/document/detail/zh/canncommercial/700/devtools/auxiliarydevtool/aoepar_16_034.html) file.| String   | -      |
| `insert_op_config_file_path` | Optional| Inserts the [AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/atc/atlasatc_16_0016.html) operator into a model.| String  | [AIPP](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha001/devaids/devtools/atc/atlasatc_16_0016.html) configuration file path|

## Runtime

After obtaining the converted model, use the matching runtime inference framework to perform inference. For details about how to use runtime to perform inference, see [Using C++ Interface to Perform Inference](https://www.mindspore.cn/lite/docs/en/r2.6.0/infer/runtime_cpp.html).

## Executinge the Benchmark

MindSpore Lite provides a benchmark test tool, which can be used to perform quantitative (performance) analysis on the execution time consumed by forward inference of the MindSpore Lite model. In addition, you can perform comparative error (accuracy) analysis based on the output of a specified model.
For details about the inference tool, see [benchmark](https://www.mindspore.cn/lite/docs/en/r2.6.0/tools/benchmark_tool.html).

- Performance analysis

    ```bash
    ./benchmark --device=Ascend310 --modelFile=./models/test_benchmark.ms --timeProfiling=true
    ```

- Accuracy analysis

    ```bash
    ./benchmark --device=Ascend310 --modelFile=./models/test_benchmark.ms --inDataFile=./input/test_benchmark.bin --inputShapes=1,32,32,1 --accuracyThreshold=3 --benchmarkDataFile=./output/test_benchmark.out
    ```

    To set environment variables, add the directory where the `so` library of `libmindspore-lite.so` (in `mindspore-lite-{version}-{os}-{arch}/runtime/lib`) is located to `${LD_LIBRARY_PATH}`.

## Advanced Features

### Dynamic Shape

The batch size is not fixed in certain scenarios. For example, in the target detection+facial recognition cascade scenario, the number of detected targets is subject to change, which means that the batch size of the targeted recognition input is dynamic. It would be a great waste of compute resources to perform inferences using the maximum batch size or image size. Thanks to Lite's support for dynamic batch size and dynamic image size on the Atlas 200/300/500 inference product, you can configure the [acl_option_cfg_param] dynamic parameter through configFile to convert a model into an `ms` model, and then use the [resize](https://www.mindspore.cn/lite/docs/en/r2.6.0/infer/runtime_cpp.html#resizing-the-input-dimension) function of the model to change the input shape during inference.

#### Dynamic Batch Size

- Parameter name

    dynamic_batch_size

- Function

    Sets the dynamic batch size parameter. This parameter applies to the scenario where the number of images to be processed each time is not fixed during inference. This parameter must be used together with input_shape_vector and cannot be used together with dynamic_image_size.

- Value

    Up to 100 batch sizes are supported. Separate batch sizes with commas (,). The value range is [1, 2048]. For example, parameters in a configuration file are set as follows:

    ```cpp
    [acl_option_cfg_param]
    input_shape_vector="[-1,32,32,4]"
    dynamic_batch_size="2,4"
    ```

    "-1" in input_shape indicates that the batch size is dynamic. The value range is "2,4". That is, size 0: [2, 32, 32, 4] and size 1: [4, 32, 32, 4] are supported.

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --inputShape="input:4,32,32,4" --configFile=./config.txt --outputFile=${model_name}
    ```

    Note: To enable the dynamic batch size function, you need to set inputShape to the shape corresponding to the maximum size (value of size 1 in the preceding example). In addition, you need to configure the dynamic batch size of [acl_option_cfg_param] through configFile (as shown in the preceding example).

- Inference

    After the dynamic batch size is enabled, during model inference, the input shape is corresponding to the size configured in converter. To change the input shape, use the model [resize](https://www.mindspore.cn/lite/docs/en/r2.6.0/infer/runtime_cpp.html#resizing-the-input-dimension) function.

- Precautions

    (1) This parameter allows you to run inference with dynamic batch sizes. For example, to run inference on two, four, or eight images per batch, set this parameter to 2,4,8. Memory will be allocated based on the runtime batch size.<br/>
    (2) Too large batch sizes or too many batch size profiles will cause model build failures.<br/>
    (3) In the scenario where you have set too large batch sizes or too many batch size profiles, you are advised to run the swapoff -a command to disable the use of swap space as memory to prevent slow running of the operating environment.<br/>

#### Dynamic Image Size

- Parameter name

    dynamic_image_size

- Function

    Sets dynamic image size profiles. This parameter applies to the scenario where the width and height of the image processed each time are not fixed during inference. This parameter must be used together with input_shape_vector and cannot be used together with dynamic_batch_size.

- Value

    A maximum of 100 image size profiles are supported. Separate image sizes with semicolons (;). The format is "imagesize1_height,imagesize1_width;imagesize2_height,imagesize2_width". Enclose all parameters in double quotation marks (""), and separate the parameters with semicolons (;). For example, parameters in a configuration file are set as follows:

    ```cpp
    [acl_option_cfg_param]
    input_format="NCHW"
    input_shape_vector="[2,3,-1,-1]"
    dynamic_image_size="64,64;96,96"
    ```

    "-1" in input_shape indicates that the image size is dynamic. That is, size 0 [2,3,64,64] and size 1 [2,3,96,96] are supported.

- converter

    ```bash
    ./converter_lite --fmk=ONNX --modelFile=${model_name}.onnx --inputShape="input:2,3,96,96" --configFile=./config.txt --outputFile=${model_name}
    ```

    Note: To enable the dynamic batch size function, you need to set inputShape to the shape corresponding to the maximum size (value of size 1 in the preceding example). In addition, you need to configure the dynamic image size of [acl_option_cfg_param] through configFile (as shown in the preceding example).

- Inference

    After the dynamic image size is enabled, during model inference, the input shape is corresponding to the size configured in converter. To change the input shape, use the model [resize](https://www.mindspore.cn/lite/docs/en/r2.6.0/infer/runtime_cpp.html#resizing-the-input-dimension) function.

- Precautions

    (1) Too large image sizes or too many image size profiles will cause model build failures.<br/>
    (2) If dynamic image size is enabled, the size of the dataset images used for inference must match the runtime image size in use.<br/>
    (3) In the scenario where you have set too large image sizes or too many image size profiles, you are advised to run the swapoff -a command to disable the use of swap space as memory to prevent slow operating environment.<br/>

## Supported Operators

For details about the supported operators, see [Lite Operator List](https://www.mindspore.cn/lite/docs/en/r2.6.0/reference/operator_list_lite.html).
