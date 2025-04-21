# Device-side Models Conversion

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/converter/converter_tool.md)

## Overview

MindSpore Lite provides a tool for offline model conversion. It supports conversion of multiple types of models. The converted models can be used for inference. The command line parameters contain multiple personalized options, providing a convenient conversion method for users.

Currently, the following input formats are supported: MindSpore, TensorFlow Lite, Caffe, TensorFlow, ONNX, and PyTorch.

The `ms` model converted by the conversion tool supports the conversion tool and the higher version of the Runtime framework to perform inference.

## Linux Environment Instructions

### Environment Preparation

To use the MindSpore Lite model conversion tool, you need to prepare the environment as follows:

- [Compile](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html) or [download](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) model transfer tool.

- Add the path of dynamic library required by the conversion tool to the environment variables LD_LIBRARY_PATH.

    ```bash
    export LD_LIBRARY_PATH=${PACKAGE_ROOT_PATH}/tools/converter/lib:${LD_LIBRARY_PATH}
    ```

    ${PACKAGE_ROOT_PATH} is the decompressed package path obtained by compiling or downloading.
- If you use Python 3.11 when compiling the MindSpore Lite package, you need to add the Python dynamic link library to the environment variable LD_LIBRARY_PATH when using the conversion tools and the runtime tools.

    ```bash
    export LD_LIBRARY_PATH=${PATHON_ROOT_PATH}/lib:${LD_LIBRARY_PATH}
    ```

  ${PATHON_ROOT_PATH} is the path of the Python environment used. After decoupling the Python dependency, the environment variable does not need to be set.

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
            ├── libmindspore_glog.so.0         # Dynamic library of Glog
            ├── libmslite_converter_plugin.so  # Dynamic library of plugin registry
            ├── libopencv_core.so.4.5          # Dynamic library of OpenCV
            ├── libopencv_imgcodecs.so.4.5     # Dynamic library of OpenCV
            └── libopencv_imgproc.so.4.5       # Dynamic library of OpenCV
```

### Parameter Description

MindSpore Lite model conversion tool provides multiple parameters.
You can enter `./converter_lite --help` to obtain the help information in real time.

The following describes the parameters in detail.

| Parameter  |  Mandatory or Not   | Parameter Description  | Value Range | Default Value | Remarks |
| -------- | ------- | ----- | --- | ---- | ---- |
| `--help` | No | Prints all the help information. | - | - | - |
| `--fmk=<FMK>`  | Yes | Original format of the input model. | MINDIR, CAFFE, TFLITE, TF, ONNX or PYTORCH, MSLITE | - | can only be set to MSLITE when it is for code generation |
| `--modelFile=<MODELFILE>` | Yes | Path of the input model. | - | - | - |
| `--outputFile=<OUTPUTFILE>` | Yes | Path of the output model. The suffix `.ms` can be automatically generated. | - | - | - |
| `--weightFile=<WEIGHTFILE>` | Yes (for Caffe models only) | Path of the weight file of the input model. | - | - | - |
| `--configFile=<CONFIGFILE>` | No | 1. Configure quantization parameter; 2. Profile path for extension. | - | - | - |
| `--fp16=<FP16>` | No | Serialize const tensor in float16 data type, only effective for const tensor in float32 data type. | on or off | off | - |
| `--inputShape=<INPUTSHAPE>` | No | Set the dimension of the model input, the order of input dimensions is consistent with the original model. For some models, the model structure can be further optimized, but the transformed model may lose the characteristics of dynamic shape. Multiple inputs are separated by `;`, and surround with `""` | e.g.  "inTensorName_1: 1,32,32,4;inTensorName_2:1,64,64,4;" | - | - |
| `--saveType=<SAVETYPE>` | No | Set the exported model as `mindir` model or `ms` model. | MINDIR, MINDIR_LITE | MINDIR_LITE | This device-side version can only be reasoned with models turned out by setting to MINDIR_LITE | - | - |
| `--optimize=<OPTIMIZE>` | No | Set the optimization accomplished in the process of converting model. | none, general, gpu_oriented, ascend_oriented | general | - | - |
| `--inputDataFormat=<INPUTDATAFORMAT>` | No | Set the input format of exported model. Only valid for 4-dimensional inputs. | NHWC, NCHW | NHWC | - |
| `--decryptKey=<DECRYPTKEY>` | No | The key used to decrypt the MindIR file, expressed in hexadecimal characters. Only valid when fmkIn is 'MINDIR'. | - | - | - |
| `--decryptMode=<DECRYPTMODE>` | No | Decryption mode for the MindIR file. Only valid when dec_key is set. | AES-GCM, AES-CBC | AES-GCM | - |
| `--inputDataType=<INPUTDATATYPE>` | No | Set data type of input tensor of quantized model. Only valid for input tensor which has quantization parameters(scale and zero point). Keep same with the data type of input tensor of origin model by default. | FLOAT32, INT8, UINT8, DEFAULT | DEFAULT | - |
| `--outputDataType=<OUTPUTDATATYPE>` | No | Set data type of output tensor of quantized model. Only valid for output tensor which has quantization parameters(scale and zero point). Keep same with the data type of output tensor of origin model by default. | FLOAT32, INT8, UINT8, DEFAULT | DEFAULT | - |
| `--outputDataFormat=<OUTPUTDATAFORMAT>` | No | Set the output format of exported model. Only valid for 4-dimensional outputs. | NHWC, NCHW | - | - |
| `--encryptKey=<ENCRYPTKEY>` | No              | Set the key for exporting encrypted `ms` models. The key is expressed in hexadecimal. Only AES-GCM is supported, and the key length is only 16Byte. | - | - | - |
| `--encryption=<ENCRYPTION>` | No | Set whether to encrypt when exporting the `ms` model. Exporting encryption can protect the integrity of the model, but it will increase the runtime initialization time. | true, false | false | - |
| `--infer=<INFER>` | No | Set whether to pre-inference when conversion is complete. | true, false | false | - |

> - The parameter name and parameter value are separated by an equal sign (=) and no space is allowed between them.
> - Because the compilation option that supports the conversion of PyTorch models is turned off by default, the downloaded installation package does not support the conversion of PyTorch models. You need to open the specified compilation option to compile locally. The following preconditions must be met for converting PyTorch models:`export MSLITE_ENABLE_CONVERT_PYTORCH_MODEL=on && export LIB_TORCH_PATH="/home/user/libtorch"` before compiling. Users can download [CPU version libtorch](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip). Then unzip it to the directory `/home/user/libtorch`. Add the environment variable of libtorch before conversion: `export LD_LIBRARY_PATH="/home/user/libtorch/lib:${LD_LIBRARY_PATH}"`.
> - The Caffe model is divided into two files: model structure `*.prototxt`, corresponding to the `--modelFile` parameter; model weight `*.caffemodel`, corresponding to the `--weightFile` parameter.
> - The priority of `--fp16` option is very low. For example, if quantization is enabled, `--fp16` will no longer take effect on const tensors that have been quantized. All in all, this option only takes effect on const tensors of float32 when serializing model.
> - `inputDataFormat`: generally, in the scenario of integrating third-party hardware of NCHW specification, designated as NCHW will have a significant performance improvement over NHWC. In other scenarios, users can also set as needed.
> - The `configFile` configuration files uses the `key=value` mode to define related parameters. For the configuration parameters related to quantization, please refer to [quantization](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/quantization.html). For the configuration parameters related to extension, please refer to [Extension Configuration](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/advanced/third_party/converter_register.html#extension-configuration).
> - `--optimize` parameter is used to set the mode of optimization during the offline conversion. If this parameter is set to none, no relevant graph optimization operations will be performed during the offline conversion phase of the model, and the relevant graph optimization operations will be done during the execution of the inference phase. The advantage of this parameter is that the converted model can be deployed directly to any CPU/GPU/Ascend hardware backend since it is not optimized in a specific way, while the disadvantage is that the initialization time of the model increases during inference execution. If this parameter is set to general, general optimization will be performed, such as constant folding and operator fusion (the converted model only supports CPU/GPU hardware backend, not Ascend backend). If this parameter is set to gpu_oriented, the general optimization and extra optimization for GPU hardware will be performed (the converted model only supports GPU hardware backend). If this parameter is set to ascend_oriented, the optimization for Ascend hardware will be performed (the converted model only supports Ascend hardware backend).
> - The encryption and decryption function only takes effect when `MSLITE_ENABLE_MODEL_ENCRYPTION=on` is set at [compile](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html) time and only supports Linux x86 platforms, and the key is a string represented by hexadecimal. For example, if the key is defined as `b'0123456789ABCDEF'`, the corresponding hexadecimal representation is `30313233343536373839414243444546`. Users on the Linux platform can use the `xxd` tool to convert the key represented by the bytes to a hexadecimal representation.
It should be noted that the encryption and decryption algorithm has been updated in version 1.7. As a result, the new version of the converter tool does not support the conversion of the encrypted model exported by MindSpore in version 1.6 and earlier.
> - Parameters `--input_shape` and dynamicDims are stored in the model during conversion. Call model.get_model_info("input_shape") and model.get_model_info("dynamic_dims") to get it when using the model.

### CPU Model Optimization

If the converted ms model is running on android cpu backend, and hope the model compile with lower latency. Try to turn on this optimization. Add the configuration item `[cpu_option_cfg_param]` in the `configFile` to get a lower compile latency model. At present, the optimization is only available when the model include Matmul operator and its data type is `float32` or dynamic quantization is enabled.

| Parameter | Attribute | Function Description | Value Range |
|--------|--------|--------|--------|
|    `architecture`    |    Mandatory    |     target cpu architecture, only support ARM64    |     ARM64    |
|    `instruction`    |    Mandatory    |    target instruction set, only support SMID_DOT    |    SIMD_DOT    |

### Example

The following describes how to use the conversion command by using several common examples.

- Take the Caffe model LeNet as an example. Run the following conversion command:

   ```bash
   ./converter_lite --fmk=CAFFE --modelFile=lenet.prototxt --weightFile=lenet.caffemodel --outputFile=lenet
   ```

   In this example, the Caffe model is used. Therefore, the model structure and model weight files are required. Two more parameters `fmk` and `outputFile` are also required.

   The output is as follows:

   ```text
   CONVERT RESULT SUCCESS:0
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

    - PyTorch model `model.pt`

      ```bash
      export LD_LIBRARY_PATH="/home/user/libtorch/lib:${LD_LIBRARY_PATH}"
      export LIB_TORCH_PATH="/home/user/libtorch"
      ./converter_lite --fmk=PYTORCH --modelFile=model.pt --outputFile=model
      ```

    - PyTorch model `model.pth`

      ```bash
      export LD_LIBRARY_PATH="/home/user/libtorch/lib:${LD_LIBRARY_PATH}"
      export LIB_TORCH_PATH="/home/user/libtorch"
      ./converter_lite --fmk=PYTORCH --modelFile=model.pth --outputFile=model
      ```

     > The following preconditions must be met for converting PyTorch models: `export MSLITE_ENABLE_CONVERT_PYTORCH_MODEL=on && export LIB_TORCH_PATH="/home/user/libtorch"` before compiling. Users can download [CPU version libtorch](https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip). Then unzip it to the directory `/home/user/libtorch`. Add the environment variable of libtorch before conversion: `export LD_LIBRARY_PATH="/home/user/libtorch/lib:${LD_LIBRARY_PATH}"`.

   In the preceding scenarios, the following information is displayed, indicating that the conversion is successful. In addition, the target file `model.ms` is obtained.

   ```text
   CONVERT RESULT SUCCESS:0
   ```

## Windows Environment Instructions

### Environment Preparation  

To use the MindSpore Lite model conversion tool, the following environment preparations are required.

- [Compile](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/build/build.html) or [download](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) model transfer tool.

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
            ├── libmindspore_glog.dll            # Dynamic library of Glog
            ├── libmslite_converter_plugin.dll   # Dynamic library of plugin registry
            ├── libmslite_converter_plugin.dll.a # Link file of Dynamic library of plugin registry
            ├── libssp-0.dll          # Dynamic library of MinGW
            ├── libstdc++-6.dll       # Dynamic library of MinGW
            └── libwinpthread-1.dll   # Dynamic library of MinGW
```

### Parameter Description

Refer to the Linux environment model conversion tool [parameter description](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/converter/converter_tool.html#parameter-description).

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
   CONVERT RESULT SUCCESS:0
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
  CONVERT RESULT SUCCESS:0
  ```
