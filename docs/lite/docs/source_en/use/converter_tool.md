# Converting Models for Inference

`Windows` `Linux` `Model Converting` `Intermediate` `Expert`

<!-- TOC -->

- [Converting Models for Inference](#converting-models-for-inference)
    - [Overview](#overview)
    - [Linux Environment Instructions](#linux-environment-instructions)
        - [Environment Preparation](#environment-preparation)
        - [Directory Structure](#directory-structure)
        - [Parameter Description](#parameter-description)
        - [Example](#example)
    - [Windows Environment Instructions](#windows-environment-instructions)
        - [Environment Preparation](#environment-preparation-1)
        - [Directory Structure](#directory-structure-1)
        - [Parameter Description](#parameter-description-1)
        - [Example](#example-1)
    - [Advanced Usage](#advanced-usage)
        - [Pass Extension](#pass-extension)
        - [Operator Infershape Extension](#operator-infershape-extension)
        - [Example](#example-2)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/lite/docs/source_en/use/converter_tool.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore Lite provides a tool for offline model conversion. It supports conversion of multiple types of models. The converted models can be used for inference. The command line parameters contain multiple personalized options, providing a convenient conversion method for users.

Currently, the following input formats are supported: MindSpore, TensorFlow Lite, Caffe, TensorFlow and ONNX.

The ms model converted by the conversion tool supports the conversion tool and the higher version of the Runtime framework to perform inference.

## Linux Environment Instructions

### Environment Preparation

To use the MindSpore Lite model conversion tool, you need to prepare the environment as follows:

- [Compile](https://www.mindspore.cn/lite/docs/en/r1.5/use/build.html) or [download](https://www.mindspore.cn/lite/docs/en/r1.5/use/downloads.html) model transfer tool.

- Add the path of dynamic library required by the conversion tool to the environment variables LD_LIBRARY_PATH.

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ````

    ${PACKAGE_ROOT_PATH} is the decompressed package path obtained by compiling or downloading.

### Directory Structure

```text
mindspore-lite-{version}-linux-x64
└── tools
    └── converter
        ├── include
        │   └── registry             # Header files of customized op, model parser, node parser and pass registration
        ├── converter                # Model conversion tool
        │   └── converter_lite       # Executable program
        └── lib                      # The dynamic link library that converter depends
            ├── libglog.so.0         # Dynamic library of Glog
            ├── libmslite_converter_plugin.so  # Dynamic library of plugin registry
            ├── libopencv_core.so.4.5          # Dynamic library of OpenCV
            ├── libopencv_imgcodecs.so.4.5     # Dynamic library of OpenCV
            └── libopencv_imgproc.so.4.5       # Dynamic library of OpenCV
```

### Parameter Description

MindSpore Lite model conversion tool provides multiple parameters.
You can enter `./converter_lite --help` to obtain the help information in real time.

The following describes the parameters in detail.

| Parameter  |  Mandatory or Not   |  Parameter Description  | Value Range | Default Value |
| -------- | ------- | ----- | --- | ---- |
| `--help` | No | Prints all the help information. | - | - |
| `--fmk=<FMK>`  | Yes | Original format of the input model. | MINDIR, CAFFE, TFLITE, TF, or ONNX | - |
| `--modelFile=<MODELFILE>` | Yes | Path of the input model. | - | - |
| `--outputFile=<OUTPUTFILE>` | Yes | Path of the output model. The suffix `.ms` can be automatically generated. | - | - |
| `--weightFile=<WEIGHTFILE>` | Yes (for Caffe models only) | Path of the weight file of the input model. | - | - |
| `--configFile=<CONFIGFILE>` | No | 1) Configure quantization parameter; 2) Profile path for extension. | - | - |
| `--fp16=<FP16>` | No | Serialize const tensor in Float16 data type, only effective for const tensor in Float32 data type. | on or off | off |
| `--inputShape=<INPUTSHAPE>` | No | Set the dimension of the model input, the order of input dimensions is consistent with the original model. For some models, the model structure can be further optimized, but the transformed model may lose the characteristics of dynamic shape. Multiple inputs are separated by `;`, and surround with `""` | e.g.  "inTensorName_1: 1,32,32,4;inTensorName_2:1,64,64,4;" | - |
| `--inputDataFormat=<INPUTDATAFORMAT>` | No | Set the input format of exported model. Only valid for 4-dimensional inputs. | NHWC, NCHW | NHWC |

> - The parameter name and parameter value are separated by an equal sign (=) and no space is allowed between them.
> - The Caffe model is divided into two files: model structure `*.prototxt`, corresponding to the `--modelFile` parameter; model weight `*.caffemodel`, corresponding to the `--weightFile` parameter.
> - The priority of `--fp16` option is very low. For example, if quantization is enabled, `--fp16` will no longer take effect on const tensors that have been quantized. All in all, this option only takes effect on const tensors of Float32 when serializing model.
> - `inputDataFormat`: generally, in the scenario of integrating third-party hardware of NCHW specification([Usage Description of the Integrated NNIE](https://www.mindspore.cn/lite/docs/en/r1.5/use/nnie.html#nnie)), designated as NCHW will have a significant performance improvement over NHWC. In other scenarios, users can also set as needed.

The calibration dataset configuration file uses the `key=value` mode to define related parameters. For the configuration parameters related to quantization, please refer to [post training quantization](https://www.mindspore.cn/lite/docs/en/r1.5/use/post_training_quantization.html). For the configuration parameters related to extension, please refer to [Extension Configuration](https://www.mindspore.cn/lite/docs/en/r1.5/use/nnie.html#extension-configuration).

| Parameter Name | Attribute | Function Description | Parameter Type | Default Value | Value Range |
| -------- | ------- | -----          | -----    | -----     |  ----- |
| plugin_path | Optional | Third-party library path | String | - | If there are more than one, please use `;` to separate. |
| disable_fusion | Optional | Indicate whether to correct the quantization error | String | off | off or on. |

### Example

The following describes how to use the conversion command by using several common examples.

- Take the Caffe model LeNet as an example. Run the following conversion command:

   ```bash
   ./converter_lite --fmk=CAFFE --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
   ```

   In this example, the Caffe model is used. Therefore, the model structure and model weight files are required. Two more parameters `fmk` and `outputFile` are also required.

   The output is as follows:

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

   This indicates that the Caffe model is successfully converted into the MindSpore Lite model and the new file `lenet.ms` is generated.

- The following uses the MindSpore, TensorFlow Lite, TensorFlow and ONNX models as examples to describe how to run the conversion command.

    - MindSpore model `model.mindir`

      ```bash
      ./converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model
      ```

     > The `MindIR` model exported by MindSpore v1.1.1 or earlier is recommended to be converted to the `ms` model using the converter tool of the corresponding version. MindSpore v1.1.1 and later versions, the converter tool will be forward compatible.

    - TensorFlow Lite model `model.tflite`

      ```bash
      ./converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model
      ```

    - TensorFlow model `model.pb`

      ```bash
      ./converter_lite --fmk=TF --modelFile=model.pb --outputFile=model
      ```

    - ONNX model `model.onnx`

      ```bash
      ./converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model
      ```

   In the preceding scenarios, the following information is displayed, indicating that the conversion is successful. In addition, the target file `model.ms` is obtained.

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

## Windows Environment Instructions

### Environment Preparation  

To use the MindSpore Lite model conversion tool, the following environment preparations are required.

- [Compile](https://www.mindspore.cn/lite/docs/en/r1.5/use/build.html) or [download](https://www.mindspore.cn/lite/docs/en/r1.5/use/downloads.html) model transfer tool.

- Add the path of dynamic library required by the conversion tool to the environment variables PATH.

    ```bash
    set PATH=%PACKAGE_ROOT_PATH%\tools\converter\lib;%PATH%
    ````

    %PACKAGE_ROOT_PATH% is the decompressed package path obtained by compiling or downloading.

### Directory Structure

```text
mindspore-lite-{version}-win-x64
└── tools
    └── converter # Model conversion tool
        ├── converter
        │   └── converter_lite.exe    # Executable program
        └── lib
            ├── libgcc_s_seh-1.dll    # Dynamic library of MinGW
            ├── libglog.dll           # Dynamic library of Glog
            ├── libmslite_converter_plugin.dll   # Dynamic library of plugin registry
            ├── libmslite_converter_plugin.dll.a # Link file of Dynamic library of plugin registry
            ├── libssp-0.dll          # Dynamic library of MinGW
            ├── libstdc++-6.dll       # Dynamic library of MinGW
            └── libwinpthread-1.dll   # Dynamic library of MinGW
```

### Parameter Description

Refer to the Linux environment model conversion tool [parameter description](https://www.mindspore.cn/lite/docs/en/r1.5/use/converter_tool.html#parameter-description).

### Example

Set the log printing level to INFO.

```bat
set GLOG_v=1
```

> Log level: 0 is DEBUG, 1 is INFO, 2 is WARNING, 3 is ERROR.

Several common examples are selected below to illustrate the use of conversion commands.

- Take the Caffe model LeNet as an example to execute the conversion command.

   ```bat
   call converter_lite --fmk=CAFFE --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
   ```

   In this example, because the Caffe model is used, two input files of model structure and model weight are required. Then with the fmk type and output path two parameters which are required, you can successfully execute.

   The result is shown as:

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

   This means that the Caffe model has been successfully converted to the MindSpore Lite model and the new file `lenet.ms` has been obtained.

- Take MindSpore, TensorFlow Lite, ONNX model format and perceptual quantization model as examples to execute conversion commands.

    - MindSpore model `model.mindir`

      ```bat
      call converter_lite --fmk=MINDIR --modelFile=model.mindir --outputFile=model
      ```

     > The `MindIR` model exported by MindSpore v1.1.1 or earlier is recommended to be converted to the `ms` model using the converter tool of the corresponding version. MindSpore v1.1.1 and later versions, the converter tool will be forward compatible.

    - TensorFlow Lite model`model.tflite`

      ```bat
      call converter_lite --fmk=TFLITE --modelFile=model.tflite --outputFile=model
      ```

    - TensorFlow model `model.pb`

      ```bat
      call converter_lite --fmk=TF --modelFile=model.pb --outputFile=model
      ```

    - ONNX model`model.onnx`

      ```bat
      call converter_lite --fmk=ONNX --modelFile=model.onnx --outputFile=model
      ```

   In the above cases, the following conversion success prompt is displayed, and the `model.ms` target file is obtained at the same time.

   ```text
   CONVERTER RESULT SUCCESS:0
   ```

## Advanced Usage

The extension ability is only supported in Linux.

In this chapter, we will show the users an example of extending Mindspore Lite converter tool, covering the whole process of creating pass, compiling and linking. The example will help the users understand the advanced usage as soon as possible.

The chapter takes a [add.tflite](https://download.mindspore.cn/model_zoo/official/lite/quick_start/add.tflite), which only includes an opreator of adding, as an example. We will show the users how to convert the single operator of adding to that of [Custom](https://www.mindspore.cn/lite/docs/en/r1.5/use/register_kernel.html#custom) and finally, obtain a model, which only includs a single operator of custom.

The code related to the example can be obtained from the directory [mindspore/lite/examples/converter_extend](https://gitee.com/mindspore/mindspore/tree/r1.5/mindspore/lite/examples/converter_extend).

### Pass Extension

1. Self-defined Pass: The users need to inherit the base class [PassBase](https://www.mindspore.cn/lite/api/en/r1.5/api_cpp/mindspore_registry.html#passbase), and override the interface function [Execute](https://www.mindspore.cn/lite/api/en/r1.5/api_cpp/mindspore_registry.html#execute)。

2. Pass Registration: The users can directly call the registration interface [REG_PASS](https://www.mindspore.cn/lite/api/en/r1.5/api_cpp/mindspore_registry.html#reg-pass), so that the self-defined pass can be registered in the converter tool of MindSpore Lite.

### Operator Infershape Extension

In the offline phase of conversion, we will infer the basic information of output tensors of each node of the model, including the format, data type and shape. So, in this phase, users need to provide the inferring process of self-defined operator. Here, users can refer to [Operator Infershape Extension](https://www.mindspore.cn/lite/docs/en/r1.5/use/runtime_cpp.html#id19)。

### Example

#### Compile

- Environment Requirements

    - System environment: Linux x86_64; Recommend Ubuntu 18.04.02LTS
    - compilation dependencies:
        - [CMake](https://cmake.org/download/) >= 3.18.3
        - [GCC](https://gcc.gnu.org/releases.html) >= 7.3.0

- Compilation and Build

  Execute the script [build.sh](https://gitee.com/mindspore/mindspore/blob/r1.5/mindspore/lite/examples/converter_extend/build.sh) in the directory of `mindspore/lite/examples/converter_extend`. And then, the released package of Mindspore Lite will be downloaded and the demo will be compiled automatically.

  ```bash
  bash build.sh
  ```

  > If the automatic download is failed, users can download the specified package manually, of which the hardware platform is CPU and the system is Ubuntu-x64 [mindspore-lite-{version}-linux-x64.tar.gz](https://www.mindspore.cn/lite/docs/en/r1.5/use/downloads.html), After unzipping, please copy the directory of `tools/converter/lib` and `tools/converter/include` to the directory of `mindspore/lite/examples/converter_extend`.
  >
  > After manually downloading and storing the specified file, users need to execute the `build.sh` script to complete the compilation and build process.

- Compilation Result

  The dynamic library `libconverter_extend_tutorial.so` will be generated in the directory of `mindspore/lite/examples/converter_extend/build`.

#### Execute Program

1. Copy library

   Copy the dynamic library `libconverter_extend_tutorial.so` to the directory of `tools/converter/lib` of the released package.

2. Enter the conversion directory of the released package.

   ```bash
   cd ${PACKAGE_ROOT_PATH}/tools/converter/converter
   ```

3. Create extension configuration file(converter.cfg), the content is as follows:

   ```text
   [registry]
   plugin_path=libconverter_extend_tutorial.so      # users need to configure the correct path of the dynamic library
   ```

4. Add the required dynamic library to the environment variable `LD_LIBRARY_PATH`

   ```bash
   export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/tools/converter/lib
   ```

5. Execute the script

   ```bash
   ./converter_lite --fmk=TFLITE --modelFile=add.tflite --configFile=converter.cfg --outputFile=add_extend
   ```

The model file `add_extend.ms` will be generated, the place of which is up to the parameter `outputFile`.
