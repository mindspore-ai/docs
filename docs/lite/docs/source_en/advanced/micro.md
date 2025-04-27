# Performing Inference or Training on MCU or Small Systems

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/advanced/micro.md)

## Overview

This tutorial describes an ultra-lightweight AI deployment solution for IoT edge devices.

Compared with mobile devices, MCUs are usually used on IoT devices. The ROM resources of the device are limited, and the memory and computing power of the hardware resources are weak.
Therefore, AI applications on IoT devices have strict limits on runtime memory and power consumption of AI model inference.
For MCUs deploying hardware backends, MindSpore Lite provides the ultra-lightweight Micro AI deployment solution. In the offline phase, models are directly generated into lightweight code without online model parsing and graph compilation. The generated Micro code is easy to understand, with less memory at runtime and smaller code size.
You can use a MindSpore Lite conversion tool `converter_lite` to easily generate inference or training code that can be deployed on the x86/ARM64/ARM32/Cortex-M platform.

Deploying a model for inference or training via the Micro involves the following four steps: model code generation, `Micro` lib obtaining, code integration, and compilation and deployment.

## Generating Model Inference Code

### Overview

The Micro configuration item in the parameter configuration file is configured via the MindSpore Lite conversion tool `convert_lite`.
This chapter describes the functions related to code generation in the conversion tool. For details about how to use the conversion tool, see [Converting Models for Inference](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html).

### Preparing Environment

The following describes how to prepare the environment for using the conversion tool in the Linux environment.

1. System environment required for running the conversion tool

    In this example, the Linux operating system is used. Ubuntu 18.04.02LTS is recommended.

2. Obtain the conversion tool

    You can obtain the conversion tool in either of the following ways:

    - Download [Release Version](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) from the MindSpore official website.

        Download the release package whose OS is Linux-x86_64 and hardware platform is CPU.

    - Start from the source code for [Building MindSpore Lite](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html).

3. Decompress the downloaded package.

    ```bash
    tar -zxf mindspore-lite-${version}-linux-x64.tar.gz
    ```

    ${version} is the version number of the release package.

4. Add the dynamic link library required by the conversion tool to the environment variable LD_LIBRARY_PATH.

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ```

    ${PACKAGE_ROOT_PATH} is the path of the decompressed folder.

### Generating Inference Code in Single Model Scenario

1. Go to the conversion directory

    ```bash
    cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
    ```

2. Set the Micro configuration item

    Create the micro.cfg file in the current directory. The file content is as follows:

    ```text
    [micro_param]

    # enable code-generation for MCU HW

    enable_micro=true

    # specify HW target, support x86,Cortex-M, AMR32A, ARM64 only.

    target=x86

    # enable parallel inference or not.

    support_parallel=false

    ```

    In the configuration file, `[micro_param]` in the first line indicates that the subsequent variable parameters belong to the micro configuration item `micro_param`. These parameters are used to control code generation. Table 1 describes the parameters.
    In this example, we will generate single model inference code for Linux systems with the underlying architecture x86_64, so set `target=x86` to declare that the generated inference code will be used for Linux systems with the underlying architecture x86_64.

3. Prepare the model to generate inference code

    Click here to download the [MNIST Handwritten Digit Recognition Model](https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/mnist.tar.gz) used in this example.
    After downloading, decompress the package to obtain `mnist.tflite`. This model is a trained MNIST classification model, that is, a TFLITE model. Copy the `mnist.tflite` model to the current conversion tool directory.

4. Execute converter_lite and generate code

    ```bash
    ./converter_lite --fmk=TFLITE --modelFile=mnist.tflite --outputFile=mnist --configFile=micro.cfg
    ```

    The following information is displayed when the code is run successfully:

    ```text
    CONVERT RESULT SUCCESS:0
    ```

    For details about the parameters related to converter_lite, see [Converter Parameter Description](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html#parameter-description).

    After the conversion tool is successfully executed, the generated code is saved in the specified `outputFile` directory. In this example, the mnist folder is in the current conversion directory. The content is as follows:

    ```text
    mnist                          # Specified name of generated code root directory
    ├── benchmark                  # Benchmark routines for integrated calls to model inference code
    │   ├── benchmark.c
    │   ├── calib_output.c
    │   ├── calib_output.h
    │   ├── load_input.c
    │   └── load_input.h
    ├── CMakeLists.txt             # cmake project file of the benchmark routine
    └── src                        # Model inference code directory
        ├── model0                 # Directory related to specify model
           ├── model0.c
           ├── net0.bin            # Model weights in binary form
           ├── net0.c
           ├── net0.h
           ├── weight0.c
           ├── weight0.h
        ├── CMakeLists.txt
        ├── allocator.c
        ├── allocator.h
        ├── net.cmake
        ├── model.c
        ├── model.h
        ├── context.c
        ├── context.h
        ├── tensor.c
        ├── tensor.h
    ```

    The `src` directory in the generated code is the directory where the model inference code is located. The `benchmark` is just a routine for calling the `src` directory code integratedly.
    For more details on integrated calls, please refer to the section on [Code Integration and Compilation Deployment](#code-integration-and-compilation-deployment).

Table 1: micro_param Parameter Definition

| Parameter          | Mandatory or not        | Parameter Description                                                                                          | Range                   | Default value    |
| --------------- |-------------------------|----------------------------------------------------------------------------------------------------------------| --------------------------| --------- |
| enable_micro    | Yes                     | The model generates code, otherwise it generates .ms.                                                          | true, false                | false      |
| target          | Yes                     | Platform for which code is generated                                                                           | x86, Cortex-M, ARM32, ARM64 | x86       |
| support_parallel | No                      | Whether to generate multi-threaded inference codes, which can be set to true only on x86/ARM32/ARM64 platforms | true, false | false       |
| save_path      | No(Multi-model param)   | The path of multi-model generated code directory                                                               |             |             |
| project_name     | No(Multi-model param)   | Multi-model generated code project name                                                                        |             |             |
| inputs_shape     | No(Dynamic shape param) | Input shape information of models in dynamic shape scenes                                                      |             |             |
|dynamic_dim_params|  No(Dynamic shape param)    | The value range of variable dimensions in dynamic shape scenes                                                 |             |             |

### Generating Inference Code in Multi-model Scenario

1. Go to the conversion directory

    ```bash
    cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
    ```

2. Set the Micro configuration item

   Create the `micro.cfg` file in the current directory. The file content is as follows:

    ```text
    [micro_param]

    # enable code-generation for MCU HW

    enable_micro=true

    # specify HW target, support x86,Cortex-M, AMR32A, ARM64 only.

    target=x86

    # enable parallel inference or not.

    support_parallel=false

    # save generated code path.

    save_path=workpath/

    # set project name.

    project_name=minst

    [model_param]

    # input model type.

    fmk=TFLITE

    # path of input model file.

    modelFile=mnist.tflite

    [model_param]

    # input model type.

    fmk=TFLITE

    # path of input model file.

    modelFile=mnist.tflite

    ```

   In the configuration file, `[micro_param]` in the first line indicates that the subsequent variable parameters belong to the micro configuration item `micro_param`. These parameters are used to control code generation, and the meaning of each parameter is shown in Table 1. `[model_param]` indicates that the subsequent variable parameters belong to the specify model configuration item`model_param`. These parameters are used to control the conversion of different models. The range of parameters includes the necessary parameters supported by `converter_lite`.
   In this example, we will generate single model inference code for Linux systems with the underlying architecture x86_64, so set `target=x86` to declare that the generated inference code will be used for Linux systems with the underlying architecture x86_64.

3. Prepare the model to generate inference code

   Click here to download the [MNIST Handwritten Digit Recognition Model](https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/mnist.tar.gz) used in this example.
   After downloading, decompress the package to obtain `mnist.tflite`. This model is a trained MNIST classification model, that is, a TFLITE model. Copy the `mnist.tflite` model to the current conversion tool directory.

4. Execute converter_lite. The user only needs to set the configFile, and then the code is generated

    ```bash
    ./converter_lite --configFile=micro.cfg
    ```

   The following information is displayed when the code is run successfully:

    ```text
    CONVERT RESULT SUCCESS:0
    ```

   For details about the parameters related to converter_lite, see [Converter Parameter Description](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html#parameter-description).

   After the conversion tool is successfully executed, the generated code is saved in the specified `save_path` + `project_name` directory. In this example, the mnist folder is in the current conversion directory. The content is as follows:

    ```text
    mnist                          # Specified name of generated code root directory
    ├── benchmark                  # Benchmark routines for integrated calls to model inference code
    │   ├── benchmark.c
    │   ├── calib_output.c
    │   ├── calib_output.h
    │   ├── load_input.c
    │   └── load_input.h
    ├── CMakeLists.txt             # cmake project file of the benchmark routine
    ├── include
        ├── model_handle.h         # Model external interface file
    └── src                        # Model inference code directory
        ├── model0                 # Directory related to specify model
           ├── model0.c
           ├── net0.bin            # Model weights in binary form
           ├── net0.c
           ├── net0.h
           ├── weight0.c
           ├── weight0.h
        ├── model1                 # Directory related to specify model
           ├── model1.c
           ├── net1.bin            # Model weights in binary form
           ├── net1.c
           ├── net1.h
           ├── weight1.c
           ├── weight1.h
        ├── CMakeLists.txt
        ├── allocator.c
        ├── allocator.h
        ├── net.cmake
        ├── model.c
        ├── model.h
        ├── context.c
        ├── context.h
        ├── tensor.c
        ├── tensor.h
    ```

   The `src` directory in the generated code is the directory where the model inference code is located. The `benchmark` is just a routine for calling the `src` directory code integratedly. In multi-model inference scenario, users need to modify the `benchmark` according to their own needs.
   For more details on integrated calls, please refer to the section on [Code Integration and Compilation Deployment](#code-integration-and-compilation-deployment).

### (Optional) Model Input Shape Configuration

Usually, when generating code, you can reduce the probability of errors in the deployment process by configuring the model input shape as the input shape for actual inference.
When the model contains a `Shape` operator or the original model has a non-fixed input shape value, the input shape value of the model must be configured to support the relevant shape optimization and code generation.
The `--inputShape=` command of the conversion tool can be used to configure the input shape of the generated code. For specific parameter meanings, please refer to [Conversion Tool Instructions](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html).

### (Optional) Dynamic Shape Configuration

In some inference scenarios, such as detecting a target and then executing the target recognition network, the number of targets is not fixed resulting in a variable input BatchSize for the target recognition network. If each inference is regenerated and deployed according to the required BatchSize or resolution, it will result in wasted memory resources and reduced development efficiency. Therefore, it needs to support dynamic shape capability for Micro. The dynamic shape parameter in `[micro_param]` is configured via configFile in the convert phase, and the [MSModelResize](#calling-interface-of-inference-code) is used during inference, to change the input shape.
Among them, all input shape information of the configuration model in `inputs_shape` is represented by real numbers for fixed dimensions and placeholders for dynamic dimensions. Currently, only two variable dimensions are supported for configuration. `dynamic_dim_params` represents the range of variable dimension values, which needs to correspond to the placeholder configured by `inputs_shape`; If the range is a discrete value, it is separated by `,`, and if the range is a continuous value, it is separated by `~`. All parameters are written in compact format without leaving any spaces in between; If there are multiple inputs, the gear corresponding to different inputs needs to be consistent, and use `;` to separate, otherwise the parsing will fail.

```text
[micro_param]

# the name and shapes of model's all inputs.
# the format is 'input1_name:[d0,d1];input2_name:[1,d0]'
inputs_shape=input1:[d0,d1];input2:[1,d0]

# the value range of dynamic dims.
dynamic_dim_params=d0:[1,3];d1:[1~8]

```

### (Optional) Generating Multithreaded Parallel Inference Code

In the usual Linux-x86/android scenario, with multi-core CPUs, Micro multi-threaded inference is enabled to leverage device performance and speed up model inference.

#### Configuration

By setting the `support_parallel` to true in the configuration file, the code supporting multi-threaded inference will be generated. Please refer to Table 1 for the meaning of each option in the configuration file.
An example of a `x86` multithreaded code generation configuration file is as follows:

```text
[micro_param]

# enable code-generation for MCU HW

enable_micro=true

# specify HW target, support x86,Cortex-M, AMR32A, ARM64 only.

target=x86

# enable parallel inference or not.

support_parallel=true

```

#### Involved Calling Interfaces

By integrating the code and calling the following interfaces, the user can configure the multi-threaded inference of the model.
For specific interface parameters, refer to [API Document](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/index.html).

Table 2: API Interface for Multi-threaded Configuration

| Function            | Function definition                                                                |
| ---------------- | ----------------------------------------------------------------------- |
| set the number of threads during inference | void MSContextSetThreadNum(MSContextHandle context, int32_t thread_num) |
| set the thread affinity to CPU cores | void MSContextSetThreadAffinityMode(MSContextHandle context, int mode)  |
| Obtain the current thread number setting during inference | int32_t MSContextGetThreadNum(const MSContextHandle context);  |
| obtain the thread affinity of CPU cores | int MSContextGetThreadAffinityMode(const MSContextHandle context)  |

#### Integration Considerations

After generating multithreaded code, users need to link to the `pthread` standard library and the `libwrapper.a` static library in the Micro library.
Please refer to the `CMakeLists.txt` file in the generated code for details.

#### Restrictions

At present, this function is only enabled when the `target` is configured as x86/ARM32/ARM64. The maximum number of inference threads can be set to 4.

### (Optional) Generating Int8 Quantitative Inference Code

In MCU scenarios such as Cortex-M, limited by the memory size and computing power of the device, Int8 quantization operators are usually used for deployment inference to reduce the runtime memory size and speed up operations.

If the user already has an Int8 full quantitative model, you can refer to the section on [Generating Inference Code by Running converter_lite](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/micro.html#generating-inference-code-by-running-converter-lite) to try to generate Int8 quantitative inference code directly without reading this chapter.
In general, the user has only one trained float32 model. To generate Int8 quantitative inference code at this time, it is necessary to cooperate with the post quantization function of the conversion tool to generate code. See the following for specific steps.

#### Configuration

Int8 quantization inference code can be generated by configuring quantization control parameters in the configuration file. For the description of quantization control parameters (`universal quantization parameters` and `full quantization parameters`), please refer to the [Quantization](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/quantization.html).

An example of Int8 quantitative inference code generation configuration file for a `Cortex-M` platform is as follows:

```text
[micro_param]
# enable code-generation for MCU HW
enable_micro=true

# specify HW target, support x86,Cortex-M, ARM32, ARM64 only.
target=Cortex-M

# code generation for Inference or Train
codegen_mode=Inference

# enable parallel inference or not
support_parallel=false

[common_quant_param]
# Supports WEIGHT_QUANT or FULL_QUANT
quant_type=FULL_QUANT

# Weight quantization support the number of bits [0,16], Set to 0 is mixed bit quantization, otherwise it is fixed bit quantization
# Full quantization support the number of bits [1,8]
bit_num=8

[data_preprocess_param]

calibrate_path=inputs:/home/input_dir

calibrate_size=100

input_type=BIN

[full_quant_param]

activation_quant_method=MAX_MIN

bias_correction=true

target_device=DSP

```

##### Restrictions

- Currently, it only supports full quantitative inference code generation.

- The `target_device` of the `full quantization parameter` in the configuration file usually needs to be set to DSP to support more operators for post quantization.

- At present, Micro has supported 8 Int8 quantization operators(add, batchnorm, concat, conv, convolution, matmul, resize, slice). If a related quantization operator does not support it when generating code, you can circumvent the operator through the `skip_quant_node` of the `universal quantization parameter`. The circumvented operator node still uses float32 inference.

## Generating Model Training Code

### Overview

The training code can be generated for the input model by using the MindSpore Lite conversion tool `converter_lite` and configuring the Micro configuration item in the parameter configuration file of the conversion tool.
This chapter describes the functions related to code generation in the conversion tool. For details about how to use the conversion tool, see [Converting Models for Training](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/train/converter_train.html).

### Preparing Environment

For preparing environment section, refer to the [above](#preparing-environment), which will not be repeated here.

### Generating Inference Code by Running converter_lite

1. Go to the conversion directory

    ```bash
    cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
    ```

2. Set the Micro configuration item

    Create the `micro.cfg` file in the current directory. The file content is as follows:

    ```text
    [micro_param]

    # enable code-generation for MCU HW

    enable_micro=true

    # specify HW target, support x86,Cortex-M, AMR32A, ARM64 only.

    target=x86

    # code generation for Inference or Train. Cortex-M is unsupported when codegen_mode is Train.

    codegen_mode=Train

    ```

3. Execute converter_lite and generate code

    ```bash
    ./converter_lite --fmk=MINDIR --trainModel=True --modelFile=my_model.mindir --outputFile=my_model --configFile=micro.cfg
    ```

    The following information is displayed when the code is run successfully:

    ```text
    CONVERT RESULT SUCCESS:0
    ```

    After the conversion tool is successfully executed, the generated code is saved in the specified `outputFile` directory. In this example, the my_model folder is in the current conversion directory. The content is as follows:

    ```text
    my_model                       # Specified name of generated code root directory
    ├── benchmark                  # Benchmark routines for integrated calls to model train code
    │   ├── benchmark.c
    │   ├── calib_output.c
    │   ├── calib_output.h
    │   ├── load_input.c
    │   └── load_input.h
    ├── CMakeLists.txt             # cmake project file of the benchmark routine
    └── src                        # Model inference code directory
        ├── CMakeLists.txt
        ├── net.bin                # Model weights in binary form
        ├── net.c
        ├── net.cmake
        ├── net.h
        ├── model.c
        ├── context.c
        ├── context.h
        ├── tensor.c
        ├── tensor.h
        ├── weight.c
        └── weight.h
    ```

    For the API involved in the training process, please refer to the [Introduction to training interface](#calling-interface-of-training-code)

## Obtaining `Micro` Lib

After generating model inference code, you need to obtain the `Micro` lib on which the generated inference code depends before performing integrated development on the code.

The inference code of different platforms depends on the `Micro` lib of the corresponding platform. You need to specify the platform via the micro configuration item `target` based on the platform in use when generating code, and obtain the `Micro` lib of the platform when obtaining the inference package.
You can download the [Release Version](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) of the corresponding platform from the MindSpore official website.

In chapter [Generating Model Inference Code](#generating-model-inference-code), we obtain the model inference code of the Linux platform with the x86_64 architecture. The `Micro` lib on which the code depends is the release package used by the conversion tool.
In the release package, the following content depended by the inference code:

```text
mindspore-lite-{version}-linux-x64
├── runtime
│   └── include
│       └── c_api            # C API header file integrated with MindSpore Lite
└── tools
    └── codegen # The source code generated by code depends on include and lib
        ├── include          # Inference framework header file
        │   ├── nnacl        # nnacl operator header file
        │   └── wrapper      # wrapper operator header file
        ├── lib
        │   ├── libwrapper.a # The MindSpore Lite codegen generates some operator static libraries on which the code depends
        │   └── libnnacl.a   # The MindSpore Lite codegen generates the nnacl operator static library on which the code depends
        └── third_party
            ├── include
            │   └── CMSIS    # ARM CMSIS NN operator header file
            └── lib
                └── libcmsis_nn.a # ARM CMSIS NN operator static library
```

## Code Integration and Compilation Deployment

In the `benchmark` directory where the code is generated, there is an interface call example for the inference code.
Users can refer to the benchmark routine to integrate and develop the `src` inference code to realize their own applications.

### Calling Interface of Inference Code

The following is the general calling interface of the inference code. For a detailed description of the interface, please refer to the [API documentation](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/index.html).

Table 3: Inference Common API Interface

| Function                                                                                       | Function definition                                                                                                                                                                    |
|------------------------------------------------------------------------------------------------| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Create a model object                                                                          | MSModelHandle MSModelCreate()                                                                                                                                               |
| Destroy the model object                                                                       | void MSModelDestroy(MSModelHandle *model)                                                                                                                                   |
| Calculate the workspace size required for model inference, valid only on cortex-M architecture | size_t MSModelCalcWorkspaceSize(MSModelHandle model)                       |
| Set workspace for the model object, valid only on cortex-M architecture                        | void MSModelSetWorkspace(MSModelHandle model, void *workspace, size_t workspace_size)                        |
| Compile model                                                                                  | MSStatus MSModelBuild(MSModelHandle model, const void *model_data, size_t data_size, MSModelType model_type, const MSContextHandle model_context)                           |
| Set the input shapes of the model                                                              | MSStatus MSModelResize(MSModelHandle model, const MSTensorHandleArray inputs, MSShapeInfo *shape_infos, size_t shape_info_num)                                                                                                                                                                            |
| Inference model                                                                                | MSStatus MSModelPredict(MSModelHandle model, const MSTensorHandleArray inputs, MSTensorHandleArray *outputs, const MSKernelCallBackC before, const MSKernelCallBackC after) |
| Obtain all input tensor handles of the model                                                   | MSTensorHandleArray MSModelGetInputs(const MSModelHandle model)                                                                                                             |
| Obtain all output tensor handles of the model                                                  | MSTensorHandleArray MSModelGetOutputs(const MSModelHandle model)                                                                                                            |
| Obtain the input tensor handle of the model by name                                            | MSTensorHandle MSModelGetInputByTensorName(const MSModelHandle model, const char *tensor_name)                                                                              |
| Obtain the output tensor handle of the model by name                                           | MSTensorHandle MSModelGetOutputByTensorName(const MSModelHandle model, const char *tensor_name)  |

### Calling Interface of Training Code

The following is the general calling interface of the Training code.

Table 4: Training Common API Interface (only training-related interfaces are listed here)

| Function                  | Function definition                                                                                                                                                                    |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Run model by step        | MSStatus MSModelRunStep(MSModelHandle model, const MSKernelCallBackC before, const MSKernelCallBackC after) |
| Set the model running mode      | MSStatus MSModelSetTrainMode(MSModelHandle model, bool train) |
| Export the weights of model to file      | MSStatus MSModelExportWeight(MSModelHandle model, const char *export_path) |

### Integration Differences of Different Platforms

Different platforms have differences in code integration and compilation deployment.

- For the MCU of the cortex-M architecture, see [Performing Inference on the MCU](#performing-inference-on-the-mcu)

- For the Linux platform with the x86_64 architecture, see [Compilation and Deployment on Linux_x86_64 Platform](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/quick_start_micro/mnist_x86)

- For details about how to compile and deploy arm32 or arm64 on the Android platform, see [Compilation and Deployment on Android Platform](https://gitee.com/mindspore/mindspore/tree/v2.6.0-rc1/mindspore/lite/examples/quick_start_micro/mobilenetv2_arm64)

- For compilation and deployment on the OpenHarmony platform, see [Executing Inference on Light Harmony Devices](#executing-inference-on-light-harmony-devices)

### Integration of Multi-model Inference Scenario

Multi-model integration is similar to single model integration. The only difference is that in the single model scenario, users can create model through the `MSModelCreate` interface. While in multi-model scenario, the `MSModeHandle` handle is provided for users. Users can integrate different models by manipulating the `MSModeHandle` handle of different models and calling the inference common API interface of single model. The `MSModeHandle` handle can refer to the `model_handle.h` file in the multi-model directory.

## Performing Inference on the MCU

### Overview

This tutorial takes the deployment of the [MNIST model](https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/mnist.tar.gz) on the STM32F767 chip as an example to demonstrate how to deploy the inference model on the MCU of the Cortex-M architecture, including the following steps:

- Use the converter_lite conversion tool to generate model inference code that adapts to the Cortex-M architecture

- Download the `Micro` lib corresponding to the Cortex-M architecture

- Integrate and compile the obtained inference code and `Micro` lib, and deploy verification

    On the Windows platform, we demonstrate how to develop inference code through IAR, while on the Linux platform, we demonstrate how to develop inference code through MakeFile cross-compilation.

### Generating MCU Inference Code

Generate inference code for the MCU. For details, see the chapter [Generating Model Inference Code](#generating-model-inference-code). You only need to change `target=x86` in the Micro configuration item to `target=Cortex-M` to generate inference code for the MCU.
After the code is generated, the contents of the folder are as follows:

```text
mnist                          # Specified name of generated code root directory
├── benchmark                  # Benchmark routines for integrated calls to model inference code
│   ├── benchmark.c
│   ├── calib_output.c
│   ├── calib_output.h
│   ├── data.c
│   ├── data.h
│   ├── load_input.c
│   └── load_input.h
├── build.sh                   # One-click compilation script
├── CMakeLists.txt             # cmake project file of the benchmark routine
├── cortex-m7.toolchain.cmake  # Cross-compilation cmake file for cortex-m7
└── src                        # Model inference code directory
    ├── CMakeLists.txt
    ├── context.c
    ├── context.h
    ├── model.c
    ├── net.c
    ├── net.cmake
    ├── net.h
    ├── tensor.c
    ├── tensor.h
    ├── weight.c
    └── weight.h
```

### Downloading `Micro` Lib of Cortex-M Architecture

The STM32F767 uses the Cortex-M7 architecture. You can obtain the `Micro` lib of the architecture in either of the following ways:

- Download [Release Version](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) from the MindSpore official website.

    You need to download the release package whose OS is None and hardware platform is Cortex-M7.

- Start from the source code for [Building MindSpore Lite](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html).

    You can run the `MSLITE_MICRO_PLATFORM=cortex-m7 bash build.sh -I x86_64` command to compile the Cortex-M7 release package.

For other Cortex-M architecture platforms that do not provide release packages for download, you can modify MindSpore source code and manually compile the code to obtain the release package by referring to the method of compiling and building from source code.

### Code Integration and Compilation Deployment on Windows: Integrated Development Through IAR

This example shows code integration and burning through IAR and demonstrates how to develop the generated inference code in Windows. The main steps are as follows:

- Download the required software and prepare the integrated environment.

- Generate the required MCU startup code and demonstration project by using the `STM32CubeMX` software.

- Integrate model inference code and `Micro` lib within `IAR`.

- Perform complilation and simulation run

#### Environment Preparation

- [STM32CubeMX Windows Version](https://www.st.com/en/development-tools/stm32cubemx.html) >= 6.0.1

    - `STM32CubeMX` is a graphical configuration tool for STM32 chips provided by `STM`. This tool is used to generate the startup code and project of STM chips.

- [IAR EWARM](https://www.iar.com/ewarm) >= 9.1

    - `IAR EWARM` is an integrated development environment developed by IAR Systems for ARM microprocessors.

#### Obtaining the MCU Startup Code and Project

If you have an MCU project, skip this chapter.
This chapter uses the STM32F767 startup project as an example to describe how to generate an MCU project for an STM32 chip via `STM32CubeMX`.

- Start `STM32CubeMX` and select `New Project` from `File` to create a project.

- In the `MCU/MPU Selector` window, search for and select `STM32F767IGT6`, and click `Start Project` to create a project for the chip

- On the `Project Manager` page, configure the project name and the path of the generated project, and select `EWARM` in `Toolchain / IDE` to generate the IAR project

- Click `GENERATE CODE` above to generate code

- On the PC where the `IAR` is installed, double-click `Project.eww` in the `EWARM` directory of the generated project to open the IAR project.

#### Integrating Model Inference Code and `Micro` Lib

- Copy the generated inference code to the project, decompress the package obtained in [Downloading `Micro` Lib of Cortex-M Architecture](#downloading-micro-lib-of-cortex-m-architecture), and place it in the generated inference code directory, as shown in the following figure:

    ```text
    test_stm32f767                                   # MCU project directory
    ├── Core
    │   ├── Inc
    │   └── Src
    │       ├── main.c
    │       └── ...
    ├── Drivers
    ├── EWARM                                        # IAR project file directory
    └── mnist                                        # Generated Code Root Directory
        ├── benchmark                                # Benchmark routines for integrated calls to model inference code
        │   ├── benchmark.c
        │   ├── data.c
        │   ├── data.h
        │   └── ...
        │── mindspore-lite-1.8.0-none-cortex-m7      # Downloaded Cortex-M7 Architecture `Micro` Lib
        ├── src                                      # Model inference code directory
        └── ...
    ```

- Import source files to the IAR project

    Open the IAR project. On the `Workspace` page, right-click the project, choose `Add > Add Group` and add a `mnist` group. Right-click the group and repeat the operations to create groups `src` and `benchmark`.
    Choose `Add -> Add Files` to import the source files in the `src` and `benchmark` directories in the `mnist` folder to the groups.

- Add the dependent header file path and static library

    On the `Workspace` page, right-click the project and choose `Options` from the shortcut menu. Select `C/C++ Compiler` in the left pane of the project options window. In the right pane, select `Preprocessor` and add the path of the header file on which the inference code depends to the list. In this example, the path of the header file is as follows:

    ```text
    $PROJ_DIR$/../mnist/mindspore-lite-1.8.0-none-cortex-m7/runtime
    $PROJ_DIR$/../mnist/mindspore-lite-1.8.0-none-cortex-m7/runtime/include
    $PROJ_DIR$/../mnist/mindspore-lite-1.8.0-none-cortex-m7/tools/codegen/include
    $PROJ_DIR$/../mnist/mindspore-lite-1.8.0-none-cortex-m7/tools/codegen/third_party/include/CMSIS/Core
    $PROJ_DIR$/../mnist/mindspore-lite-1.8.0-none-cortex-m7/tools/codegen/third_party/include/CMSIS/DSP
    $PROJ_DIR$/../mnist/mindspore-lite-1.8.0-none-cortex-m7/tools/codegen/third_party/include/CMSIS/NN
    $PROJ_DIR$/../mnist
    ```

    In the left pane of the Project Options window, select `Linker`. In the right pane, select `Library` and add the operator static library file on which the inference code depends to the list. The static library file added in this example is as follows:

    ```text
    $PROJ_DIR$/../mnist/mindspore-lite-1.8.0-none-cortex-m7/tools/codegen/lib/libwrapper.a
    $PROJ_DIR$/../mnist/mindspore-lite-1.8.0-none-cortex-m7/tools/codegen/lib/libnnacl.a
    $PROJ_DIR$/../mnist/mindspore-lite-1.8.0-none-cortex-m7/tools/codegen/third_party/lib/libcmsis_nn.a  
    ```

- Modify the main.c file and invoke the benchmark function

    Add a header file reference at the beginning of `main.c` and invoke the `benchmark` function in `benchmark.c` in the main function. The program in the benchmark folder is a sample program that invokes the inference code in the generated `src` and compares the output, which can be modified freely.

    ```c++
    #include "benchmark/benchmark.h"
    ...
    int main(void)
    {
      ...
      if (benchmark() == 0) {
          printf("\nrun success.\n");
      } else {
          printf("\nrun failed.\n");
      }
      ...
    }
    ```

- Modify the `mnist/benchmark/data.c` file to store benchmark input and output data in the program for comparison

    In the benchmark routine, the input data of the model is set and the inference result is compared with the expected result to obtain the error offset.
    In this example, set the input data of the model by modifying the `calib_input0_data` array of `data.c`, and set the expected result by modifying the `calib_output0_data`.

    ```c++
    float calib_input0_data[NET_INPUT0_SIZE] = {0.54881352186203,0.7151893377304077,0.6027633547782898,0.5448831915855408,0.42365479469299316,0.6458941102027893,0.4375872015953064,0.891772985458374,0.9636627435684204,0.3834415078163147,0.7917250394821167,0.5288949012756348,0.5680445432662964,0.9255966544151306,0.07103605568408966,0.08712930232286453,0.020218396559357643,0.832619845867157,0.7781567573547363,0.8700121641159058,0.978618323802948,0.7991585731506348,0.4614793658256531,0.7805292010307312,0.11827442795038223,0.6399210095405579,0.14335328340530396,0.9446688890457153,0.5218483209609985,0.4146619439125061,0.26455560326576233,0.7742336988449097,0.4561503231525421,0.568433940410614,0.018789799883961678,0.6176354885101318,0.6120957136154175,0.6169340014457703,0.9437480568885803,0.681820273399353,0.35950788855552673,0.43703195452690125,0.6976311802864075,0.0602254718542099,0.6667667031288147,0.670637845993042,0.21038256585597992,0.12892629206180573,0.31542834639549255,0.36371076107025146,0.5701967477798462,0.4386015236377716,0.9883738160133362,0.10204481333494186,0.20887675881385803,0.16130951046943665,0.6531082987785339,0.25329160690307617,0.4663107693195343,0.24442559480667114,0.15896958112716675,0.11037514358758926,0.6563295722007751,0.13818295300006866,0.1965823620557785,0.3687251806259155,0.8209932446479797,0.09710127860307693,0.8379449248313904,0.0960984081029892,0.9764594435691833,0.4686512053012848,0.9767611026763916,0.6048455238342285,0.7392635941505432,0.03918779268860817,0.28280696272850037,0.12019655853509903,0.296140193939209,0.11872772127389908,0.3179831802845001,0.414262980222702,0.06414749473333359,0.6924721002578735,0.5666014552116394,0.26538950204849243,0.5232480764389038,0.09394051134586334,0.5759465098381042,0.9292961955070496,0.3185689449310303,0.6674103736877441,0.13179786503314972,0.7163271903991699,0.28940609097480774,0.18319135904312134,0.5865129232406616,0.02010754682123661,0.8289400339126587,0.004695476032793522,0.6778165102005005,0.2700079679489136,0.7351940274238586,0.9621885418891907,0.2487531453371048,0.5761573314666748,0.5920419096946716,0.5722519159317017,0.22308163344860077,0.9527490139007568,0.4471253752708435,0.8464086651802063,0.6994792819023132,0.2974369525909424,0.8137978315353394,0.396505743265152,0.8811032176017761,0.5812729001045227,0.8817353844642639,0.6925315856933594,0.7252542972564697,0.5013243556022644,0.9560836553573608,0.6439902186393738,0.4238550364971161,0.6063932180404663,0.019193198531866074,0.30157482624053955,0.6601735353469849,0.2900775969028473,0.6180154085159302,0.42876869440078735,0.1354740709066391,0.29828232526779175,0.5699648857116699,0.5908727645874023,0.5743252635002136,0.6532008051872253,0.6521032452583313,0.43141844868659973,0.8965466022491455,0.36756187677383423,0.4358649253845215,0.8919233679771423,0.806194007396698,0.7038885951042175,0.10022688657045364,0.9194825887680054,0.7142413258552551,0.9988470077514648,0.14944830536842346,0.8681260347366333,0.16249293088912964,0.6155595779418945,0.1238199844956398,0.8480082154273987,0.8073189854621887,0.5691007375717163,0.40718328952789307,0.06916699558496475,0.6974287629127502,0.45354267954826355,0.7220556139945984,0.8663823008537292,0.9755215048789978,0.855803370475769,0.011714084073901176,0.359978049993515,0.729990541934967,0.17162968218326569,0.5210366249084473,0.054337989538908005,0.19999653100967407,0.01852179504930973,0.793697714805603,0.2239246815443039,0.3453516662120819,0.9280812740325928,0.704414427280426,0.031838931143283844,0.1646941602230072,0.6214783787727356,0.5772286057472229,0.23789282143115997,0.9342139959335327,0.6139659285545349,0.5356327891349792,0.5899099707603455,0.7301220297813416,0.31194499135017395,0.39822107553482056,0.20984375476837158,0.18619300425052643,0.9443724155426025,0.739550769329071,0.49045881628990173,0.22741462290287018,0.2543564736843109,0.058029159903526306,0.43441662192344666,0.3117958903312683,0.6963434815406799,0.37775182723999023,0.1796036809682846,0.024678727611899376,0.06724963337182999,0.6793927550315857,0.4536968469619751,0.5365791916847229,0.8966712951660156,0.990338921546936,0.21689698100090027,0.6630781888961792,0.2633223831653595,0.02065099962055683,0.7583786249160767,0.32001715898513794,0.38346388936042786,0.5883170962333679,0.8310484290122986,0.6289818286895752,0.872650682926178,0.27354204654693604,0.7980468273162842,0.18563593924045563,0.9527916312217712,0.6874882578849792,0.21550767123699188,0.9473705887794495,0.7308558225631714,0.2539416551589966,0.21331197023391724,0.518200695514679,0.02566271834075451,0.20747007429599762,0.4246854782104492,0.3741699755191803,0.46357542276382446,0.27762871980667114,0.5867843627929688,0.8638556003570557,0.11753185838460922,0.517379105091095,0.13206811249256134,0.7168596982955933,0.39605969190597534,0.5654212832450867,0.1832798421382904,0.14484776556491852,0.4880562722682953,0.35561272501945496,0.9404319524765015,0.7653252482414246,0.748663604259491,0.9037197232246399,0.08342243731021881,0.5521924495697021,0.5844760537147522,0.961936354637146,0.29214751720428467,0.24082878232002258,0.10029394179582596,0.016429629176855087,0.9295293092727661,0.669916570186615,0.7851529121398926,0.28173011541366577,0.5864101648330688,0.06395526975393295,0.48562759160995483,0.9774951338768005,0.8765052556991577,0.3381589651107788,0.961570143699646,0.23170162737369537,0.9493188261985779,0.9413776993751526,0.799202561378479,0.6304479241371155,0.8742879629135132,0.2930202782154083,0.8489435315132141,0.6178767085075378,0.013236857950687408,0.34723350405693054,0.14814086258411407,0.9818294048309326,0.4783703088760376,0.49739137291908264,0.6394725441932678,0.36858460307121277,0.13690027594566345,0.8221177458763123,0.1898479163646698,0.5113189816474915,0.2243170291185379,0.09784448146820068,0.8621914982795715,0.9729194641113281,0.9608346819877625,0.9065554738044739,0.774047315120697,0.3331451416015625,0.08110138773918152,0.40724116563796997,0.2322341352701187,0.13248763978481293,0.053427182137966156,0.7255943417549133,0.011427458375692368,0.7705807685852051,0.14694663882255554,0.07952208071947098,0.08960303664207458,0.6720477938652039,0.24536721408367157,0.4205394685268402,0.557368814945221,0.8605511784553528,0.7270442843437195,0.2703278958797455,0.131482794880867,0.05537432059645653,0.3015986382961273,0.2621181607246399,0.45614057779312134,0.6832813620567322,0.6956254243850708,0.28351885080337524,0.3799269497394562,0.18115095794200897,0.7885454893112183,0.05684807524085045,0.6969972252845764,0.7786954045295715,0.7774075865745544,0.25942257046699524,0.3738131523132324,0.5875996351242065,0.27282190322875977,0.3708527982234955,0.19705428183078766,0.4598558843135834,0.044612299650907516,0.7997958660125732,0.07695644348859787,0.5188351273536682,0.3068101108074188,0.5775429606437683,0.9594333171844482,0.6455702185630798,0.03536243736743927,0.4304024279117584,0.5100168585777283,0.5361775159835815,0.6813924908638,0.2775960862636566,0.12886056303977966,0.3926756680011749,0.9564056992530823,0.1871308982372284,0.9039839506149292,0.5438059568405151,0.4569114148616791,0.8820413947105408,0.45860394835472107,0.7241676449775696,0.3990253210067749,0.9040443897247314,0.6900250315666199,0.6996220350265503,0.32772040367126465,0.7567786574363708,0.6360610723495483,0.2400202751159668,0.16053882241249084,0.796391487121582,0.9591665863990784,0.4581388235092163,0.5909841656684875,0.8577226400375366,0.45722344517707825,0.9518744945526123,0.5757511854171753,0.8207671046257019,0.9088436961174011,0.8155238032341003,0.15941447019577026,0.6288984417915344,0.39843425154685974,0.06271295249462128,0.4240322411060333,0.25868406891822815,0.849038302898407,0.03330462798476219,0.9589827060699463,0.35536885261535645,0.3567068874835968,0.01632850244641304,0.18523232638835907,0.40125951170921326,0.9292914271354675,0.0996149331331253,0.9453015327453613,0.869488537311554,0.4541623890399933,0.326700896024704,0.23274412751197815,0.6144647002220154,0.03307459130883217,0.015606064349412918,0.428795725107193,0.06807407736778259,0.2519409954547882,0.2211609184741974,0.253191202878952,0.13105523586273193,0.012036222964525223,0.11548429727554321,0.6184802651405334,0.9742562174797058,0.9903450012207031,0.40905410051345825,0.1629544198513031,0.6387617588043213,0.4903053343296051,0.9894098043441772,0.06530420482158661,0.7832344174385071,0.28839850425720215,0.24141861498355865,0.6625045537948608,0.24606318771839142,0.6658591032028198,0.5173085331916809,0.4240889847278595,0.5546877980232239,0.2870515286922455,0.7065746784210205,0.414856880903244,0.3605455458164215,0.8286569118499756,0.9249669313430786,0.04600730910897255,0.2326269894838333,0.34851935505867004,0.8149664998054504,0.9854914546012878,0.9689717292785645,0.904948353767395,0.2965562641620636,0.9920112490653992,0.24942004680633545,0.10590615123510361,0.9509525895118713,0.2334202527999878,0.6897682547569275,0.05835635960102081,0.7307090759277344,0.8817201852798462,0.27243688702583313,0.3790569007396698,0.3742961883544922,0.7487882375717163,0.2378072440624237,0.17185309529304504,0.4492916464805603,0.30446839332580566,0.8391891121864319,0.23774182796478271,0.5023894309997559,0.9425836205482483,0.6339976787567139,0.8672894239425659,0.940209686756134,0.7507648468017578,0.6995750665664673,0.9679655432701111,0.9944007992744446,0.4518216848373413,0.07086978107690811,0.29279401898384094,0.15235470235347748,0.41748636960983276,0.13128933310508728,0.6041178107261658,0.38280805945396423,0.8953858613967896,0.96779465675354,0.5468848943710327,0.2748235762119293,0.5922304391860962,0.8967611789703369,0.40673333406448364,0.5520782470703125,0.2716527581214905,0.4554441571235657,0.4017135500907898,0.24841345846652985,0.5058664083480835,0.31038081645965576,0.37303486466407776,0.5249704718589783,0.7505950331687927,0.3335074782371521,0.9241587519645691,0.8623185753822327,0.048690296709537506,0.2536425292491913,0.4461355209350586,0.10462789237499237,0.34847599267959595,0.7400975227355957,0.6805144548416138,0.6223844289779663,0.7105283737182617,0.20492368936538696,0.3416981101036072,0.676242470741272,0.879234790802002,0.5436780452728271,0.2826996445655823,0.030235258862376213,0.7103368043899536,0.007884103804826736,0.37267908453941345,0.5305371880531311,0.922111451625824,0.08949454873800278,0.40594232082366943,0.024313200265169144,0.3426109850406647,0.6222310662269592,0.2790679335594177,0.2097499519586563,0.11570323258638382,0.5771402716636658,0.6952700018882751,0.6719571352005005,0.9488610029220581,0.002703213831409812,0.6471966505050659,0.60039222240448,0.5887396335601807,0.9627703428268433,0.016871673986315727,0.6964824199676514,0.8136786222457886,0.5098071694374084,0.33396488428115845,0.7908401489257812,0.09724292904138565,0.44203564524650574,0.5199523568153381,0.6939564347267151,0.09088572859764099,0.2277594953775406,0.4103015661239624,0.6232946515083313,0.8869608044624329,0.618826150894165,0.13346147537231445,0.9805801510810852,0.8717857599258423,0.5027207732200623,0.9223479628562927,0.5413808226585388,0.9233060479164124,0.8298973441123962,0.968286395072937,0.919782817363739,0.03603381663560867,0.1747720092535019,0.3891346752643585,0.9521427154541016,0.300028920173645,0.16046763956546783,0.8863046765327454,0.4463944137096405,0.9078755974769592,0.16023047268390656,0.6611174941062927,0.4402637481689453,0.07648676633834839,0.6964631676673889,0.2473987489938736,0.03961552307009697,0.05994429811835289,0.06107853725552559,0.9077329635620117,0.7398838996887207,0.8980623483657837,0.6725823283195496,0.5289399027824402,0.30444636940956116,0.997962236404419,0.36218905448913574,0.47064894437789917,0.37824517488479614,0.979526937007904,0.1746583878993988,0.32798799872398376,0.6803486943244934,0.06320761889219284,0.60724937915802,0.47764649987220764,0.2839999794960022,0.2384132742881775,0.5145127177238464,0.36792758107185364,0.4565199017524719,0.3374773859977722,0.9704936742782593,0.13343943655490875,0.09680395573377609,0.3433917164802551,0.5910269021987915,0.6591764688491821,0.3972567617893219,0.9992780089378357,0.35189300775527954,0.7214066386222839,0.6375827193260193,0.8130538463592529,0.9762256741523743,0.8897936344146729,0.7645619511604309,0.6982485055923462,0.335498183965683,0.14768557250499725,0.06263600289821625,0.2419017106294632,0.432281494140625,0.521996259689331,0.7730835676193237,0.9587409496307373,0.1173204779624939,0.10700414329767227,0.5896947383880615,0.7453980445861816,0.848150372505188,0.9358320832252502,0.9834262132644653,0.39980170130729675,0.3803351819515228,0.14780867099761963,0.6849344372749329,0.6567619442939758,0.8620625734329224,0.09725799411535263,0.49777689576148987,0.5810819268226624,0.2415570467710495,0.16902540624141693,0.8595808148384094,0.05853492394089699,0.47062090039253235,0.11583399772644043,0.45705875754356384,0.9799623489379883,0.4237063527107239,0.857124924659729,0.11731556057929993,0.2712520658969879,0.40379273891448975,0.39981213212013245,0.6713835000991821,0.3447181284427643,0.713766872882843,0.6391869187355042,0.399161159992218,0.43176013231277466,0.614527702331543,0.0700421929359436,0.8224067091941833,0.65342116355896,0.7263424396514893,0.5369229912757874,0.11047711223363876,0.4050356149673462,0.40537357330322266,0.3210429847240448,0.029950324445962906,0.73725426197052,0.10978446155786514,0.6063081622123718,0.7032175064086914,0.6347863078117371,0.95914226770401,0.10329815745353699,0.8671671748161316,0.02919023483991623,0.534916877746582,0.4042436182498932,0.5241838693618774,0.36509987711906433,0.19056691229343414,0.01912289671599865,0.5181497931480408,0.8427768349647522,0.3732159435749054,0.2228638231754303,0.080532006919384,0.0853109210729599,0.22139644622802734,0.10001406073570251,0.26503971219062805,0.06614946573972702,0.06560486555099487,0.8562761545181274,0.1621202677488327,0.5596824288368225,0.7734555602073669,0.4564095735549927,0.15336887538433075,0.19959613680839539,0.43298420310020447,0.52823406457901,0.3494403064250946,0.7814795970916748,0.7510216236114502,0.9272118210792542,0.028952548280358315,0.8956912755966187,0.39256879687309265,0.8783724904060364,0.690784752368927,0.987348735332489,0.7592824697494507,0.3645446300506592,0.5010631680488586,0.37638914585113525,0.364911824464798,0.2609044909477234,0.49597030878067017,0.6817399263381958,0.27734026312828064,0.5243797898292542,0.117380291223526,0.1598452925682068,0.04680635407567024,0.9707314372062683,0.0038603513967245817,0.17857997119426727,0.6128667593002319,0.08136960119009018,0.8818964958190918,0.7196201682090759,0.9663899540901184,0.5076355338096619,0.3004036843776703,0.549500584602356,0.9308187365531921,0.5207614302635193,0.2672070264816284,0.8773987889289856,0.3719187378883362,0.0013833499979227781,0.2476850152015686,0.31823351979255676,0.8587774634361267,0.4585031569004059,0.4445872902870178,0.33610227704048157,0.880678117275238,0.9450267553329468,0.9918903112411499,0.3767412602901459,0.9661474227905273,0.7918795943260193,0.675689160823822,0.24488948285579681,0.21645726263523102,0.1660478264093399,0.9227566123008728,0.2940766513347626,0.4530942440032959,0.49395784735679626,0.7781715989112854,0.8442349433898926,0.1390727013349533,0.4269043505191803,0.842854917049408,0.8180332779884338};
    float calib_output0_data[NET_OUTPUT0_SIZE] = {3.5647096e-05,6.824297e-08,0.009327697,3.2340475e-05,1.1117579e-05,1.5117058e-06,4.6314454e-07,5.161628e-11,0.9905911,3.8835238e-10};
    ```

#### Compiling and Simulation Run

In this example, software simulation is used to view and analyze the inference result.
On the `Workspace` page, right-click the project and choose `Options` from the shortcut menu. Select the `Debugger` option on the left of the Project Options window. In the `Setup` on the right, set `Driver` as `Simulator` to enable software simulation.

Close the project option window and choose `Project > Download and Debug` on the menu bar to compile and simulate the project. By adding a breakpoint at the `benchmark`, you can observe the inference result of the simulation run and the return value of the benchmark() function.

### Code Integration and Compilation Deployment on Linux: Code Integration via MakeFile

This chapter describes how to integrate and develop the MCU inference code on the Linux platform, by taking the generated model code integration and developing through MakeFile on the Linux platform as an example.
The main steps are as follows:

- Download the required software and prepare the cross compilation and burning environment

- Generate the required MCU startup code and demonstration project using the `STM32CubeMX` software

- Modify the inference code and `Micro` lib of the `MakeFile` integrated model

- Compiling and burning the project

- Read and verify running result of the board

For the complete demo code built in this example, click [Download here](https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/test_stmf767.tar.gz).

#### Environment Preparation

- [CMake](https://cmake.org/download/) >= 3.18.3

- [GNU Arm Embedded Toolchain](https://developer.arm.com/downloads/-/gnu-rm)  >= 10-2020-q4-major-x86_64-linux

    - This tool is a cross compilation tool for Cortex-M Linux.
    - Download the `gcc-arm-none-eabi` package of the x86_64-Linux version, decompress the package, and add the bin path in the directory to the PATH environment variable: `export PATH=gcc-arm-none-eabi path/bin:$PATH`.

- [STM32CubeMX-Lin](https://www.st.com/en/development-tools/stm32cubemx.html) >= 6.5.0

    - `STM32CubeMX` is a graphical configuration tool for STM32 chips provided by `STM`, which is used to generate the startup code and project of STM chips.

- [STM32CubePrg-Lin](https://www.st.com/en/development-tools/stm32cubeprog.html) >= 6.5.0

    - This tool is a burning tool provided by `STM` and can be used for program burning and data reading.

#### Obtaining the MCU Startup Code and Project

If you have an MCU project, skip this chapter.
This chapter uses the STM32F767 startup project as an example to describe how to generate an MCU project for an STM32 chip via `STM32CubeMX`.

- Start `STM32CubeMX` and select `New Project` from `File` to create a project

- In the `MCU/MPU Selector` window, search for and select `STM32F767IGT6`, and click `Start Project` to create a project for the chip

- On the `Project Manager` page, configure the project name and the path of the generated project, and select `EWARM` in `Toolchain / IDE` to generate the IAR project

- Click `GENERATE CODE` above to generate code

- Execute `make` in the generated project directory to test if the code compiles successfully.

#### Integrating Model Inference Code and `Micro` Lib

- Copy the generated inference code to the project, decompress the package obtained in [Downloading `Micro` Lib of Cortex-M Architecture](#downloading-micro-lib-of-cortex-m-architecture), and place it in the generated inference code directory, as shown in the following figure:

    ```text
    stm32f767                                       # MCU project directory
    ├── Core
    │   ├── Inc
    │   └── Src
    │       ├── main.c
    │       └── ...
    ├── Drivers
    ├── mnist                                        # Generate Code Root Directory
    │   ├── benchmark                                # Benchmark routines for integrated calls to model inference code
    │   │   ├── benchmark.c
    │   │   ├── data.c
    │   │   ├── data.h
    │   │   └── ...
    │   │── mindspore-lite-1.8.0-none-cortex-m7      # Downloaded Cortex-M7 Architecture `Micro` Lib
    │   ├── src                                      # Model inference code directory
    │   └── ...
    ├── Makefile
    ├── startup_stm32f767xx.s
    └── STM32F767IGTx_FLASH.ld
    ```

- Modify `MakeFile` and add the model inference code and dependency library to the project

    In this example, the source code to be added to the project includes the model inference code in the `src` directory and the example code called by the model inference in the `benchmark` directory.
    Modify the definition of the `C_SOURCES` variable in `MakeFile` and add the source file path.

    ```bash
    C_SOURCES =  \
    mnist/src/context.c \
    mnist/src/model.c \
    mnist/src/net.c \
    mnist/src/tensor.c \
    mnist/src/weight.c \
    mnist/benchmark/benchmark.c \
    mnist/benchmark/calib_output.c \
    mnist/benchmark/load_input.c \
    mnist/benchmark/data.c \
    ...
    ```

    Add the path of the dependent header file: Modify the definition of the `C_INCLUDES` variable in `MakeFile` and add the following path:

    ```text
    LITE_PACK = mindspore-lite-1.8.0-none-cortex-m7

    C_INCLUDES =  \
    -Imnist/$(LITE_PACK)/runtime \
    -Imnist/$(LITE_PACK)/runtime/include \
    -Imnist/$(LITE_PACK)/tools/codegen/include \
    -Imnist/$(LITE_PACK)/tools/codegen/third_party/include/CMSIS/Core \
    -Imnist/$(LITE_PACK)/tools/codegen/third_party/include/CMSIS/DSP \
    -Imnist/$(LITE_PACK)/tools/codegen/third_party/include/CMSIS/NN \
    -Imnist \
    ...
    ```

    Add the dependent operator library (`-lnnacl -lwrapper -lcmsis_nn`), declare the path of the operator library file, and add the compilation option (`-specs=nosys.specs`).
    In this example, the modified variables are defined as follows:

    ```text
    LIBS = -lc -lm -lnosys -lnnacl -lwrapper -lcmsis_nn
    LIBDIR = -Lmnist/$(LITE_PACK)/tools/codegen/lib -Lmnist/$(LITE_PACK)/tools/codegen/third_party/lib
    LDFLAGS = $(MCU) -specs=nosys.specs -specs=nano.specs -T$(LDSCRIPT) $(LIBDIR) $(LIBS) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref -Wl,--gc-sections
    ```

- Modify the main.c file and invoke the benchmark function

    Invoke the `benchmark` function in `benchmark.c` in the main function. The program in the benchmark folder is a sample program that invokes the inference code in the generated `src` and compares the output, which can be modified freely. In this example, we call the `benchmark` function directly and assign the `run_dnn_flag` variable based on the returned result.

    ```c++
    run_dnn_flag = '0';
    if (benchmark() == 0) {
        printf("\nrun success.\n");
        run_dnn_flag = '1';
    } else {
        printf("\nrun failed.\n");
        run_dnn_flag = '2';
    }
    ```

    Add the header file reference and the definition of the `run_dnn_flag` variable to the beginning of `main.c`.

    ```c++
    #include "benchmark/benchmark.h"

    char run_dnn_flag __attribute__((section(".myram"))) ;// Array for testing
    ```

    In this example, to facilitate reading the inference result by using the burner, variables are defined in a customized section (`myram`). You can set the customized section in the following way or ignore the declaration: obtaining the inference result through serial ports or other interactive modes.

    To set a customized section, perform the following steps:
    Modify the `MEMORY` section in the `STM32F767IGTx_FLASH.ld` file, and add a customized memory segment `MYRAM`. (In this example, add 4 to the `RAM` memory start address to free up memory for `MYRAM`). Then add a customized `myram` segment declaration to the `SectionS` segment.

    ```text
    MEMORY
    {
    MYRAM (xrw)     : ORIGIN = 0x20000000, LENGTH = 1
    RAM (xrw)      : ORIGIN = 0x20000004, LENGTH = 524284
    ...
    }
    ...
    SECTIONS
    {
      ...
      .myram (NOLOAD):
      {
        . = ALIGN(4);
        _smyram = .;        /* create a global symbol at data start */
        *(.sram)           /* .data sections */
        *(.sram*)          /* .data* sections */

        . = ALIGN(4);
        _emyram = .;        /* define a global symbol at data end */
      } >MYRAM AT> FLASH
    }
    ```

- Modify the `mnist/benchmark/data.c` file to store benchmark input and output data in the program for comparison.

    In the benchmark routine, the input data of the model is set and the inference result is compared with the expected result to obtain the error offset.
    In this example, modify the `calib_input0_data` array of `data.c` to set the input data of the model, and modify the `calib_output0_data` to set the expected result.

    ```c++
    float calib_input0_data[NET_INPUT0_SIZE] = {0.54881352186203,0.7151893377304077,0.6027633547782898,0.5448831915855408,0.42365479469299316,0.6458941102027893,0.4375872015953064,0.891772985458374,0.9636627435684204,0.3834415078163147,0.7917250394821167,0.5288949012756348,0.5680445432662964,0.9255966544151306,0.07103605568408966,0.08712930232286453,0.020218396559357643,0.832619845867157,0.7781567573547363,0.8700121641159058,0.978618323802948,0.7991585731506348,0.4614793658256531,0.7805292010307312,0.11827442795038223,0.6399210095405579,0.14335328340530396,0.9446688890457153,0.5218483209609985,0.4146619439125061,0.26455560326576233,0.7742336988449097,0.4561503231525421,0.568433940410614,0.018789799883961678,0.6176354885101318,0.6120957136154175,0.6169340014457703,0.9437480568885803,0.681820273399353,0.35950788855552673,0.43703195452690125,0.6976311802864075,0.0602254718542099,0.6667667031288147,0.670637845993042,0.21038256585597992,0.12892629206180573,0.31542834639549255,0.36371076107025146,0.5701967477798462,0.4386015236377716,0.9883738160133362,0.10204481333494186,0.20887675881385803,0.16130951046943665,0.6531082987785339,0.25329160690307617,0.4663107693195343,0.24442559480667114,0.15896958112716675,0.11037514358758926,0.6563295722007751,0.13818295300006866,0.1965823620557785,0.3687251806259155,0.8209932446479797,0.09710127860307693,0.8379449248313904,0.0960984081029892,0.9764594435691833,0.4686512053012848,0.9767611026763916,0.6048455238342285,0.7392635941505432,0.03918779268860817,0.28280696272850037,0.12019655853509903,0.296140193939209,0.11872772127389908,0.3179831802845001,0.414262980222702,0.06414749473333359,0.6924721002578735,0.5666014552116394,0.26538950204849243,0.5232480764389038,0.09394051134586334,0.5759465098381042,0.9292961955070496,0.3185689449310303,0.6674103736877441,0.13179786503314972,0.7163271903991699,0.28940609097480774,0.18319135904312134,0.5865129232406616,0.02010754682123661,0.8289400339126587,0.004695476032793522,0.6778165102005005,0.2700079679489136,0.7351940274238586,0.9621885418891907,0.2487531453371048,0.5761573314666748,0.5920419096946716,0.5722519159317017,0.22308163344860077,0.9527490139007568,0.4471253752708435,0.8464086651802063,0.6994792819023132,0.2974369525909424,0.8137978315353394,0.396505743265152,0.8811032176017761,0.5812729001045227,0.8817353844642639,0.6925315856933594,0.7252542972564697,0.5013243556022644,0.9560836553573608,0.6439902186393738,0.4238550364971161,0.6063932180404663,0.019193198531866074,0.30157482624053955,0.6601735353469849,0.2900775969028473,0.6180154085159302,0.42876869440078735,0.1354740709066391,0.29828232526779175,0.5699648857116699,0.5908727645874023,0.5743252635002136,0.6532008051872253,0.6521032452583313,0.43141844868659973,0.8965466022491455,0.36756187677383423,0.4358649253845215,0.8919233679771423,0.806194007396698,0.7038885951042175,0.10022688657045364,0.9194825887680054,0.7142413258552551,0.9988470077514648,0.14944830536842346,0.8681260347366333,0.16249293088912964,0.6155595779418945,0.1238199844956398,0.8480082154273987,0.8073189854621887,0.5691007375717163,0.40718328952789307,0.06916699558496475,0.6974287629127502,0.45354267954826355,0.7220556139945984,0.8663823008537292,0.9755215048789978,0.855803370475769,0.011714084073901176,0.359978049993515,0.729990541934967,0.17162968218326569,0.5210366249084473,0.054337989538908005,0.19999653100967407,0.01852179504930973,0.793697714805603,0.2239246815443039,0.3453516662120819,0.9280812740325928,0.704414427280426,0.031838931143283844,0.1646941602230072,0.6214783787727356,0.5772286057472229,0.23789282143115997,0.9342139959335327,0.6139659285545349,0.5356327891349792,0.5899099707603455,0.7301220297813416,0.31194499135017395,0.39822107553482056,0.20984375476837158,0.18619300425052643,0.9443724155426025,0.739550769329071,0.49045881628990173,0.22741462290287018,0.2543564736843109,0.058029159903526306,0.43441662192344666,0.3117958903312683,0.6963434815406799,0.37775182723999023,0.1796036809682846,0.024678727611899376,0.06724963337182999,0.6793927550315857,0.4536968469619751,0.5365791916847229,0.8966712951660156,0.990338921546936,0.21689698100090027,0.6630781888961792,0.2633223831653595,0.02065099962055683,0.7583786249160767,0.32001715898513794,0.38346388936042786,0.5883170962333679,0.8310484290122986,0.6289818286895752,0.872650682926178,0.27354204654693604,0.7980468273162842,0.18563593924045563,0.9527916312217712,0.6874882578849792,0.21550767123699188,0.9473705887794495,0.7308558225631714,0.2539416551589966,0.21331197023391724,0.518200695514679,0.02566271834075451,0.20747007429599762,0.4246854782104492,0.3741699755191803,0.46357542276382446,0.27762871980667114,0.5867843627929688,0.8638556003570557,0.11753185838460922,0.517379105091095,0.13206811249256134,0.7168596982955933,0.39605969190597534,0.5654212832450867,0.1832798421382904,0.14484776556491852,0.4880562722682953,0.35561272501945496,0.9404319524765015,0.7653252482414246,0.748663604259491,0.9037197232246399,0.08342243731021881,0.5521924495697021,0.5844760537147522,0.961936354637146,0.29214751720428467,0.24082878232002258,0.10029394179582596,0.016429629176855087,0.9295293092727661,0.669916570186615,0.7851529121398926,0.28173011541366577,0.5864101648330688,0.06395526975393295,0.48562759160995483,0.9774951338768005,0.8765052556991577,0.3381589651107788,0.961570143699646,0.23170162737369537,0.9493188261985779,0.9413776993751526,0.799202561378479,0.6304479241371155,0.8742879629135132,0.2930202782154083,0.8489435315132141,0.6178767085075378,0.013236857950687408,0.34723350405693054,0.14814086258411407,0.9818294048309326,0.4783703088760376,0.49739137291908264,0.6394725441932678,0.36858460307121277,0.13690027594566345,0.8221177458763123,0.1898479163646698,0.5113189816474915,0.2243170291185379,0.09784448146820068,0.8621914982795715,0.9729194641113281,0.9608346819877625,0.9065554738044739,0.774047315120697,0.3331451416015625,0.08110138773918152,0.40724116563796997,0.2322341352701187,0.13248763978481293,0.053427182137966156,0.7255943417549133,0.011427458375692368,0.7705807685852051,0.14694663882255554,0.07952208071947098,0.08960303664207458,0.6720477938652039,0.24536721408367157,0.4205394685268402,0.557368814945221,0.8605511784553528,0.7270442843437195,0.2703278958797455,0.131482794880867,0.05537432059645653,0.3015986382961273,0.2621181607246399,0.45614057779312134,0.6832813620567322,0.6956254243850708,0.28351885080337524,0.3799269497394562,0.18115095794200897,0.7885454893112183,0.05684807524085045,0.6969972252845764,0.7786954045295715,0.7774075865745544,0.25942257046699524,0.3738131523132324,0.5875996351242065,0.27282190322875977,0.3708527982234955,0.19705428183078766,0.4598558843135834,0.044612299650907516,0.7997958660125732,0.07695644348859787,0.5188351273536682,0.3068101108074188,0.5775429606437683,0.9594333171844482,0.6455702185630798,0.03536243736743927,0.4304024279117584,0.5100168585777283,0.5361775159835815,0.6813924908638,0.2775960862636566,0.12886056303977966,0.3926756680011749,0.9564056992530823,0.1871308982372284,0.9039839506149292,0.5438059568405151,0.4569114148616791,0.8820413947105408,0.45860394835472107,0.7241676449775696,0.3990253210067749,0.9040443897247314,0.6900250315666199,0.6996220350265503,0.32772040367126465,0.7567786574363708,0.6360610723495483,0.2400202751159668,0.16053882241249084,0.796391487121582,0.9591665863990784,0.4581388235092163,0.5909841656684875,0.8577226400375366,0.45722344517707825,0.9518744945526123,0.5757511854171753,0.8207671046257019,0.9088436961174011,0.8155238032341003,0.15941447019577026,0.6288984417915344,0.39843425154685974,0.06271295249462128,0.4240322411060333,0.25868406891822815,0.849038302898407,0.03330462798476219,0.9589827060699463,0.35536885261535645,0.3567068874835968,0.01632850244641304,0.18523232638835907,0.40125951170921326,0.9292914271354675,0.0996149331331253,0.9453015327453613,0.869488537311554,0.4541623890399933,0.326700896024704,0.23274412751197815,0.6144647002220154,0.03307459130883217,0.015606064349412918,0.428795725107193,0.06807407736778259,0.2519409954547882,0.2211609184741974,0.253191202878952,0.13105523586273193,0.012036222964525223,0.11548429727554321,0.6184802651405334,0.9742562174797058,0.9903450012207031,0.40905410051345825,0.1629544198513031,0.6387617588043213,0.4903053343296051,0.9894098043441772,0.06530420482158661,0.7832344174385071,0.28839850425720215,0.24141861498355865,0.6625045537948608,0.24606318771839142,0.6658591032028198,0.5173085331916809,0.4240889847278595,0.5546877980232239,0.2870515286922455,0.7065746784210205,0.414856880903244,0.3605455458164215,0.8286569118499756,0.9249669313430786,0.04600730910897255,0.2326269894838333,0.34851935505867004,0.8149664998054504,0.9854914546012878,0.9689717292785645,0.904948353767395,0.2965562641620636,0.9920112490653992,0.24942004680633545,0.10590615123510361,0.9509525895118713,0.2334202527999878,0.6897682547569275,0.05835635960102081,0.7307090759277344,0.8817201852798462,0.27243688702583313,0.3790569007396698,0.3742961883544922,0.7487882375717163,0.2378072440624237,0.17185309529304504,0.4492916464805603,0.30446839332580566,0.8391891121864319,0.23774182796478271,0.5023894309997559,0.9425836205482483,0.6339976787567139,0.8672894239425659,0.940209686756134,0.7507648468017578,0.6995750665664673,0.9679655432701111,0.9944007992744446,0.4518216848373413,0.07086978107690811,0.29279401898384094,0.15235470235347748,0.41748636960983276,0.13128933310508728,0.6041178107261658,0.38280805945396423,0.8953858613967896,0.96779465675354,0.5468848943710327,0.2748235762119293,0.5922304391860962,0.8967611789703369,0.40673333406448364,0.5520782470703125,0.2716527581214905,0.4554441571235657,0.4017135500907898,0.24841345846652985,0.5058664083480835,0.31038081645965576,0.37303486466407776,0.5249704718589783,0.7505950331687927,0.3335074782371521,0.9241587519645691,0.8623185753822327,0.048690296709537506,0.2536425292491913,0.4461355209350586,0.10462789237499237,0.34847599267959595,0.7400975227355957,0.6805144548416138,0.6223844289779663,0.7105283737182617,0.20492368936538696,0.3416981101036072,0.676242470741272,0.879234790802002,0.5436780452728271,0.2826996445655823,0.030235258862376213,0.7103368043899536,0.007884103804826736,0.37267908453941345,0.5305371880531311,0.922111451625824,0.08949454873800278,0.40594232082366943,0.024313200265169144,0.3426109850406647,0.6222310662269592,0.2790679335594177,0.2097499519586563,0.11570323258638382,0.5771402716636658,0.6952700018882751,0.6719571352005005,0.9488610029220581,0.002703213831409812,0.6471966505050659,0.60039222240448,0.5887396335601807,0.9627703428268433,0.016871673986315727,0.6964824199676514,0.8136786222457886,0.5098071694374084,0.33396488428115845,0.7908401489257812,0.09724292904138565,0.44203564524650574,0.5199523568153381,0.6939564347267151,0.09088572859764099,0.2277594953775406,0.4103015661239624,0.6232946515083313,0.8869608044624329,0.618826150894165,0.13346147537231445,0.9805801510810852,0.8717857599258423,0.5027207732200623,0.9223479628562927,0.5413808226585388,0.9233060479164124,0.8298973441123962,0.968286395072937,0.919782817363739,0.03603381663560867,0.1747720092535019,0.3891346752643585,0.9521427154541016,0.300028920173645,0.16046763956546783,0.8863046765327454,0.4463944137096405,0.9078755974769592,0.16023047268390656,0.6611174941062927,0.4402637481689453,0.07648676633834839,0.6964631676673889,0.2473987489938736,0.03961552307009697,0.05994429811835289,0.06107853725552559,0.9077329635620117,0.7398838996887207,0.8980623483657837,0.6725823283195496,0.5289399027824402,0.30444636940956116,0.997962236404419,0.36218905448913574,0.47064894437789917,0.37824517488479614,0.979526937007904,0.1746583878993988,0.32798799872398376,0.6803486943244934,0.06320761889219284,0.60724937915802,0.47764649987220764,0.2839999794960022,0.2384132742881775,0.5145127177238464,0.36792758107185364,0.4565199017524719,0.3374773859977722,0.9704936742782593,0.13343943655490875,0.09680395573377609,0.3433917164802551,0.5910269021987915,0.6591764688491821,0.3972567617893219,0.9992780089378357,0.35189300775527954,0.7214066386222839,0.6375827193260193,0.8130538463592529,0.9762256741523743,0.8897936344146729,0.7645619511604309,0.6982485055923462,0.335498183965683,0.14768557250499725,0.06263600289821625,0.2419017106294632,0.432281494140625,0.521996259689331,0.7730835676193237,0.9587409496307373,0.1173204779624939,0.10700414329767227,0.5896947383880615,0.7453980445861816,0.848150372505188,0.9358320832252502,0.9834262132644653,0.39980170130729675,0.3803351819515228,0.14780867099761963,0.6849344372749329,0.6567619442939758,0.8620625734329224,0.09725799411535263,0.49777689576148987,0.5810819268226624,0.2415570467710495,0.16902540624141693,0.8595808148384094,0.05853492394089699,0.47062090039253235,0.11583399772644043,0.45705875754356384,0.9799623489379883,0.4237063527107239,0.857124924659729,0.11731556057929993,0.2712520658969879,0.40379273891448975,0.39981213212013245,0.6713835000991821,0.3447181284427643,0.713766872882843,0.6391869187355042,0.399161159992218,0.43176013231277466,0.614527702331543,0.0700421929359436,0.8224067091941833,0.65342116355896,0.7263424396514893,0.5369229912757874,0.11047711223363876,0.4050356149673462,0.40537357330322266,0.3210429847240448,0.029950324445962906,0.73725426197052,0.10978446155786514,0.6063081622123718,0.7032175064086914,0.6347863078117371,0.95914226770401,0.10329815745353699,0.8671671748161316,0.02919023483991623,0.534916877746582,0.4042436182498932,0.5241838693618774,0.36509987711906433,0.19056691229343414,0.01912289671599865,0.5181497931480408,0.8427768349647522,0.3732159435749054,0.2228638231754303,0.080532006919384,0.0853109210729599,0.22139644622802734,0.10001406073570251,0.26503971219062805,0.06614946573972702,0.06560486555099487,0.8562761545181274,0.1621202677488327,0.5596824288368225,0.7734555602073669,0.4564095735549927,0.15336887538433075,0.19959613680839539,0.43298420310020447,0.52823406457901,0.3494403064250946,0.7814795970916748,0.7510216236114502,0.9272118210792542,0.028952548280358315,0.8956912755966187,0.39256879687309265,0.8783724904060364,0.690784752368927,0.987348735332489,0.7592824697494507,0.3645446300506592,0.5010631680488586,0.37638914585113525,0.364911824464798,0.2609044909477234,0.49597030878067017,0.6817399263381958,0.27734026312828064,0.5243797898292542,0.117380291223526,0.1598452925682068,0.04680635407567024,0.9707314372062683,0.0038603513967245817,0.17857997119426727,0.6128667593002319,0.08136960119009018,0.8818964958190918,0.7196201682090759,0.9663899540901184,0.5076355338096619,0.3004036843776703,0.549500584602356,0.9308187365531921,0.5207614302635193,0.2672070264816284,0.8773987889289856,0.3719187378883362,0.0013833499979227781,0.2476850152015686,0.31823351979255676,0.8587774634361267,0.4585031569004059,0.4445872902870178,0.33610227704048157,0.880678117275238,0.9450267553329468,0.9918903112411499,0.3767412602901459,0.9661474227905273,0.7918795943260193,0.675689160823822,0.24488948285579681,0.21645726263523102,0.1660478264093399,0.9227566123008728,0.2940766513347626,0.4530942440032959,0.49395784735679626,0.7781715989112854,0.8442349433898926,0.1390727013349533,0.4269043505191803,0.842854917049408,0.8180332779884338};
    float calib_output0_data[NET_OUTPUT0_SIZE] = {3.5647096e-05,6.824297e-08,0.009327697,3.2340475e-05,1.1117579e-05,1.5117058e-06,4.6314454e-07,5.161628e-11,0.9905911,3.8835238e-10};
    ```

#### Compiling and Burning the Project

- Compile

    In the MCU project directory, run the `make` command to compile the MCU project. After the compilation is successful, the following information is displayed, in which test_stm767 is the MCU project name in this example.

    ```text
    arm-none-eabi-size build/test_stm767.elf
    text      data    bss    dec       hex      filename
    120316    3620    87885  211821    33b6d    build/test_stm767.elf
    arm-none-eabi-objcopy -O ihex build/test_stm767.elf build/test_stm767.hex
    arm-none-eabi-objcopy -O binary -S build/test_stm767.elf build/test_stm767.bin
    ```

- Burn and run

    The `STMSTM32CubePrg` tool can be used to burn and run the code. On the PC, use `STLink` to connect to a development board that can be burnt. Then, run the following commands in the current MCU project directory to burn and run the program:

    ```bash
    bash ${STMSTM32CubePrg_PATH}/bin/STM32_Programmer.sh -c port=SWD -w build/test_stm767.bin 0x08000000 -s 0x08000000
    ```

    ${STMSTM32CubePrg_PATH is}: installation path of `STMSTM32CubePrg`. For details about the parameters in the command, see the `STMSTM32CubePrg` user manual.

#### Inference Result Verification

In this example, the benchmark running result flag is stored in the memory segment whose start address is `0x20000000` and whose size is 1 byte. Therefore, you can directly obtain the data at this address by using the burner to obtain the result returned by the program.
On the PC, use `STLink` to connect to a development board where programs have been burnt. Run the following command to read the memory data:

```bash
bash ${STMSTM32CubePrg_PATH is }/bin/STM32_Programmer.sh -c port=SWD model=HOTPLUG --upload 0x20000000 0x1 ret.bin
```

${STMSTM32CubePrg_PATH is}: installation path of `STMSTM32CubePrg`. For details about the parameters in the command, see the `STMSTM32CubePrg` user manual.

The read data is saved in the `ret.bin` file and run `cat ret.bin`. If the board inference is successful and `ret.bin` stores `1`, the following information is displayed:

```text
1
```

## Executing Inference on Light Harmony Devices

### Preparing the Light Harmony Compilation Environment

You can learn how to compile and burn in the light Harmony environment at the [OpenHarmony official website](https://www.openharmony.cn).
This tutorial uses the Hi3516 development board as an example to demonstrate how to use Micro to deploy the inference model in the light Harmony environment.

### Compiling Models

Use converter_lite to compile the [lenet model](https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/mnist.tar.gz) and generate the inference code corresponding to the light Harmony platform. The command is as follows:

```shell
./converter_lite --fmk=TFLITE --modelFile=mnist.tflite --outputFile=${SOURCE_CODE_DIR} --configFile=${COFIG_FILE}
```

Set target to ARM32 in the config configuration file.

### Compiling the Build Script

For details about how to develop light Harmony applications, see [Running Hello OHOS](https://device.harmonyos.com/cn/docs/start/introduce/quickstart-lite-steps-board3516-running-0000001151888681). Copy the mnist directory generated in the previous step to any Harmony source code path, such as, applications/sample/, and create the Build.gn file.

```text
<harmony-source-path>/applications/sample/mnist
├── benchmark
├── CMakeLists.txt
├── BUILD.gn
└── src  
```

Download the [precompiled inference runtime package](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) for OpenHarmony and decompress it to any Harmony source code path. Compile Build.gn file:

```text
import("//build/lite/config/component/lite_component.gni")
import("//build/lite/ndk/ndk.gni")

lite_component("mnist_benchmark") {
    target_type = "executable"
    sources = [
        "benchmark/benchmark.cc",
        "benchmark/calib_output.cc",
        "benchmark/load_input.c",
        "src/net.c",
        "src/weight.c",
        "src/session.cc",
        "src/tensor.cc",
    ]
    features = []
    include_dirs = [
        "<YOUR MINDSPORE LITE RUNTIME PATH>/runtime",
        "<YOUR MINDSPORE LITE RUNTIME PATH>/tools/codegen/include",
        "//applications/sample/mnist/benchmark",
        "//applications/sample/mnist/src",
    ]
    ldflags = [
        "-fno-strict-aliasing",
        "-Wall",
        "-pedantic",
        "-std=gnu99",
    ]
    libs = [
        "<YOUR MINDSPORE LITE RUNTIME PATH>/runtime/lib/libmindspore-lite.a",
        "<YOUR MINDSPORE LITE RUNTIME PATH>/tools/codegen/lib/libwrapper.a",
    ]
    defines = [
        "NOT_USE_STL",
        "ENABLE_NEON",
        "ENABLE_ARM",
        "ENABLE_ARM32"
    ]
    cflags = [
        "-fno-strict-aliasing",
        "-Wall",
        "-pedantic",
        "-std=gnu99",
    ]
    cflags_cc = [
        "-fno-strict-aliasing",
        "-Wall",
        "-pedantic",
        "-std=c++17",
    ]
}
```

`<YOUR MINDSPORE LITE RUNTIME PATH>` is the path of the decompressed inference runtime package, such as //applications/sample/mnist/mindspore-lite-1.3.0-ohos-aarch32.
Modify the build/lite/components/applications.json file and add the mnist_benchmark configuration:

```text
{
    "component": "mnist_benchmark",
    "description": "Communication related samples.",
    "optional": "true",
    "dirs": [
    "applications/sample/mnist"
    ],
    "targets": [
    "//applications/sample/mnist:mnist_benchmark"
    ],
    "rom": "",
    "ram": "",
    "output": [],
    "adapted_kernel": [ "liteos_a" ],
    "features": [],
    "deps": {
    "components": [],
    "third_party": []
    }
},
```

Modify vendor/hisilicon/hispark_taurus/config.json file and add mnist_benchmark component.

```text
{ "component": "mnist_benchmark", "features":[] }
```

### Compiling benchmark

```text
cd <openharmony-source-path>
hb set(Set the compilation path)
.(Select Current Path)
Select ipcamera_hispark_taurus@hisilicon and press Enter.
hb build mnist_benchmark (Perform compilation)
```

Generate the result file out/hispark_taurus/ipcamera_hispark_taurus/bin/mnist_benchmark.

### Performing benchmark

Decompress mnist_benchmark, weight file (mnist/src/net.bin), and [input file](https://download.mindspore.cn/model_zoo/official/lite/quick_start/micro/mnist.tar.gz), copy them to the development board, and run the following commands:

```text
OHOS # ./mnist_benchmark mnist_input.bin net.bin 1
OHOS # =======run benchmark======
input 0: mnist_input.bin

loop count: 1
total time: 10.11800ms, per time: 10.11800ms

outputs:
name: int8toft32_Softmax-7_post0/output-0, DataType: 43, Elements: 10, Shape: [1 10 ], Data:
0.000000, 0.000000, 0.003906, 0.000000, 0.000000, 0.992188, 0.000000, 0.000000, 0.000000, 0.000000,
========run success=======
```

## Custom Kernel

Please refer to [Custom Kernel](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/register.html) to understand the basic concepts before using.
Micro currently only supports the registration and implementation of custom operators of custom type, and does not support the registration and custom implementation of built-in operators (such as conv2d and fc).
We use Hi3516D board as an example to show you how to use kernel register in Micro.

The manner that the model generates code is consistent with that of the non-custom operator model.

```shell
./converter_lite --fmk=TFLITE --modelFile=mnist.tflite --outputFile=${SOURCE_CODE_DIR} --configFile=${COFIG_FILE}
```

where target sets to be ARM32.

### Implementing custom kernel by users

The previous step generates the source code directory under the specified path with a header file called `src/registered_kernel.h` that specifies the function declarations for the custom operator.

``` C++
int CustomKernel(TensorC *inputs, int input_num, TensorC *outputs, int output_num, CustomParameter *param);
```

Users need to implement this function and add their source files to the cmake project. For example, we provide the custom kernel example dynamic library libmicro_nnie.so that supports NNIE from Hysis, which is included in the [official download page](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) "NNIE inference runtime lib, benchmark tool" component. Users need to modify the CMakeLists.txt of the generated code, add the name and path of the linked library.

``` shell

link_directories(<YOUR_PATH>/mindspore-lite-1.8.1-linux-aarch32/providers/Hi3516D)

link_directories(<HI3516D_SDK_PATH>)

target_link_libraries(benchmark net micro_nnie nnie mpi VoiceEngine upvqe dnvqe securec -lm -pthread)

```

In the generated `benchmark/benchmark.c` file, add the [NNIE device related initialization code](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/test/config_level0/micro/svp_sys_init.c) before and after calling the main function.
Finally, we compile the source code:

``` shell

mkdir buid && cd build

cmake -DCMAKE_TOOLCHAIN_FILE=<MS_SRC_PATH>/mindspore/lite/cmake/himix200.toolchain.cmake -DPLATFORM_ARM32=ON -DPKG_PATH=<RUNTIME_PKG_PATH> ..

make

```

## Combination of Micro Inference and Device-side Training

### Overview

Except for MCU, micro inference is a inference model that separates model structure and weight. Training usually changes its weights, but does not change its structure. So, in the scenario of combining training and inference, we can adopt the mode of device-side training plus with micro inference to take advantage of the small memory and low power consumption of micro inference. The process includes the following steps:

- Export inference model based on device-side training

- Use the converter_lite conversion tool to generate model inference code that adapts to the training architecture

- Download the Micro lib corresponding to the training architecture

- Integrate and compile the obtained inference code and Micro lib, and deploy

- Export the weights of the inference model based on device-side training, then overwrite the original weight file, and verify it

    Next, we will provide a detailed introduction to eace step and its precautions

### Exporting Inference Model

Users can directly refer to [Device-side training](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/train/runtime_train_cpp.html).

### Generating Inference Code

Users can directly refer to the above content, but two points need to be noted. Firstly, the trained model is an `ms` mode, so when using converter_lite conversion tool, the `fmk` need to be set to `MSLITE`. Secondly, in order to combine training with micro inference, it is necessary to ensure that the weights exported from training must match exactly with that from micro. Therefore, we have added two attributes to the micro configuration parameters to ensure consistency in weights.

```text
[micro_param]
# false indicates that only the required weights will be saved. Default is false.
# If collaborate with lite-train, the parameter must be true.
keep_original_weight=false

# the names of those weight-tensors whose shape is changeable, only embedding-table supports change now.
# the parameter is used to collaborate with lite-train. If set, `keep_original_weight` must be true.
changeable_weights_name=name0,name1
```

`keep_original_weight` is a key attribute that ensures consistency in weight, and when combined with training, the attribute must be set `true`. `changeable_weights_name` is used for special scenarios, such as changes in the shape of certain weights. Of course, currently only the number of embedding-table can be changeable. Generally, users do not need to set the attribute.

### Compilation and Deployment

Users can directly refer to the above content.

### Export weights of inference model

MindSpore `Serialization` class provides the `ExportWeightsCollaborateWithMicro` function, and `ExportWeightsCollaborateWithMicro` is as follows.

```cpp
  static Status ExportWeightsCollaborateWithMicro(const Model &model, ModelType model_type,
                                                  const std::string &weight_file, bool is_inference = true,
                                                  bool enable_fp16 = false,
                                                  const std::vector<std::string> &changeable_weights_name = {});
```

Here, `is_inference` currently only supports as `true`.
