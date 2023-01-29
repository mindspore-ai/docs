# Using Python Interface for Model Conversion

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/lite/docs/source_en/use/converter_python.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore Lite supports model conversion via Python interface, supporting multiple types of model conversion, and the converted models can be used for inference. The interface contains a variety of personalized parameters to provide a convenient conversion path for users. This tutorial describes how to use the [Python interface](https://www.mindspore.cn/lite/api/en/r2.0.0-alpha/mindspore_lite/mindspore_lite.Converter.html) for model conversion.

The currently supported input model types are MindSpore, TensorFlow Lite, Caffe, TensorFlow, ONNX, and PyTorch.

Convert to MindSpore Lite or MindSpore model by conversion tool when the input model type is not MindSpore. In addition, support conversion of MindSpore model to MindSpore Lite models. For inference on the generated models, the required version of Runtime inference framework is the version that comes with the conversion tool and higher version.

## Linux Environment Usage Descriptions

### Environment Preparation

To use Python interface of MindSpore Lite for model conversion, the following environment preparation is required.

- [Compile](https://www.mindspore.cn/lite/docs/en/r2.0.0-alpha/use/build.html) or [download](https://www.mindspore.cn/lite/docs/en/r2.0.0-alpha/use/downloads.html) Whl installation package of MindSpore Lite with the Converter component.

  > Currently, the installation package corresponding to Python 3.7 is available for download. If you need other Python versions, please use the compilation function to generate the installation package.

- Then use the `pip install` command to install. After installation, you can use the following command to check if the installation is successful. If no error is reported, the installation is successful.

    ```bash
    python -c "import mindspore_lite"
    ```

### Directory Structure

After successful installation, you can use the `pip show mindspore_lite` command to see where the Python modules of MindSpore Lite are installed.

```text
mindspore_lite
├── __pycache__
├── include
├── lib
│   ├── _c_lite_wrapper.cpython-37m-x86_64-linux-gnu.so         # MindSpore Lite Python module encapsulates the dynamic library of the C++ interface framework
│   ├── libmindspore_converter.so                               # Dynamic library for MindSpore Lite conversion framework
│   ├── libmindspore_core.so                                    # Dynamic library for the MindSpore Lite core framework
│   ├── libmindspore_glog.so.0                                  # Dynamic library of Glog
│   ├── libmindspore-lite.so                                    # Dynamic library for MindSpore Lite reasoning framework
│   ├── libmindspore-lite-train.so                              # Dynamic library for MindSpore Lite training framework
│   ├── libmslite_converter_plugin.so                           # Registering dynamic library for plugins
│   ├── libopencv_core.so.4.5                                   # Dynamic library of OpenCV
│   ├── libopencv_imgcodecs.so.4.5                              # Dynamic library of OpenCV
│   └── libopencv_imgproc.so.4.5                                # Dynamic library of OpenCV
├── __init__.py        # Initialization package
├── context.py         # Code related to context interface
├── converter.py       # Code related to converter interface, conversion portal
├── model.py           # Code related to model interface, inference portal
├── tensor.py          # Code related to tensor interface
└── version.py         # MindSpore Lite version number
```

### Parameter Description

Python interface model conversion of MindSpore Lite provides a variety of parameter settings that users can choose to use according to their needs.

Usage Scenarios: 1. Converting third-party models to generate MindSpore models or MindSpore Lite models, 2. Convert MindSpore models to generate MindSpore Lite models.

Detailed descriptions of the parameters and their correspondence to the parameters in [Inference Model Offline Conversion](https://www.mindspore.cn/lite/docs/en/r2.0.0-alpha/use/converter_tool.html) are provided below.

| Python interface model conversion parameters | Parameter types  | Parameters corresponding to the offline conversion of the model  |  Required or not   |  Parameters descriptions  | Range of values | Default Values |
| -------- | ----- | -------- | ------- | ----- | --- | ---- |
| fmk_type | FmkType | `--fmk=<FMK>`  | Required | The input model frame type. | FmkType.TF, FmkType.CAFFE, FmkType.ONNX, FmkType.MINDIR, FmkType.TFLITE, FmkType.PYTORCH | - |
| model_file | str | `--modelFile=<MODELFILE>` | Required | The path of the input model file for the conversion. | - | - |
| output_file | str | `--outputFile=<OUTPUTFILE>` | Required | The path to the output model when conversion can be automatically generated with a `.ms` suffix. | - | - |
| weight_file | str | `--weightFile=<WEIGHTFILE>` | Required when converting Caffe models | The path to the input model weights file. | - | "" |
| config_file | str | `--configFile=<CONFIGFILE>` | Not required | Converter profile path to configure post-training quantization or offline splitting of parallel operators, or to disable the operator fusion function and set the plug-in to the so path. | - | "" |
| weight_fp16 | bool | `--fp16=<FP16>` | Not required | Set whether the weights in Float32 data format need to be stored in Float16 data format during model serialization. | True, False | False |
| input_shape | dict{string:list\[int]} | `--inputShape=<INPUTSHAPE>` | Not required | Set the dimensions of the model input, and keep the order of the input dimensions the same as the original model. For example {"inTensor1": \[1, 32, 32, 32], "inTensor2": \[1, 1, 32, 32]} | - | None, None is equal to {} |
| input_format | Format | `--inputDataFormat=<INPUTDATAFORMAT>` | Not required | Set the input format of the exported model, valid only for 4-dimensional inputs. | Format.NCHW, Format.NHWC | Format.NHWC |
| input_data_type | DataType | `--inputDataType=<INPUTDATATYPE>` | Not required | Set the data type of the quantized model input Tensor. Only valid if the quantization parameters (`scale` and `zero point`) of the model input Tensor are available. The default is to keep the same data type as the original model input Tensor. | DataType.FLOAT32, DataType.INT8, DataType.UINT8, DataType.UNKNOWN | DataType.FLOAT32 |
| output_data_type | DataType | `--outputDataType=<OUTPUTDATATYPE>` | Not required | Set the data type of the output Tensor of the quantized model, only if the quantization parameters (`scale` and `zero point`) of the output Tensor of the model are available. The default is the same as the data type of the original model output Tensor. | DataType.FLOAT32, DataType.INT8, DataType.UINT8, DataType.UNKNOWN | DataType.FLOAT32 |
| export_mindir | ModelType | `--exportMindIR=<EXPORTMINDIR>` | Not required | Set the type of the exported model file. | ModelType.MINDIR, ModelType.MINDIR_LITE | ModelType.MINDIR_LITE |
| decrypt_key | str | `--decryptKey=<DECRYPTKEY>` | Not required | Set the key used to load the cipher text MindIR. The key is expressed in hexadecimal and is only valid when `fmk_type` is MINDIR. | - | "" |
| decrypt_mode | str | `--decryptMode=<DECRYPTMODE>` | Not required | Set the mode to load cipher MindIR, only valid when `decrypt_key` is specified. | "AES-GCM", "AES-CBC" | "AES-GCM" |
| enable_encryption | bool | `--encryption=<ENCRYPTION>` | Not required | When exporting, whether the model is encrypted. Exporting encryption protects model integrity, but increases runtime initialization time. | True, False | False |
| encrypt_key | str | `--encryptKey=<ENCRYPTKEY>` | Not required | Set the key used to encrypt the file, expressed in hexadecimal characters. Only supported when `decrypt_mode` is "AES-GCM" and the key length is 16. | - | "" |
| infer | bool | `--infer=<INFER>` | Not required | Whether to perform pre-inference at the completion of the conversion. | True, False | False |
| train_model | bool | `--trainModel=<TRAINMODEL>` | Not required | Whether the model will be trained on the device. | True, False | False |
| no_fusion | bool | `--NoFusion=<NOFUSION>` | Not required | Whether to avoid fusion optimization, the default allows fusion optimization. | True, False | False |

> For more information about the `fmk_type` parameter, see [FmkType](https://mindspore.cn/lite/api/en/r2.0.0-alpha/mindspore_lite/mindspore_lite.FmkType.html).
>
> The download installeration package does not support converting PyTorch models because the compilation option that supports converting PyTorch models is turned off by default. You need to turn on the specified compilation options locally to compile the installation package that supports converting PyTorch models. Converting the PyTorch model has the following prerequisites: before compiling, export MSLITE_ENABLE_CONVERT_PYTORCH_MODEL=on is needed, and add libtorch environment variable: export LD_LIBRARY_PATH="/home/user/libtorch/lib:${LD_LIBRARY_PATH}" && export LIB_TORCH_PATH="/home/user/libtorch" before conversion. Users can download the [CPU version libtorch](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip) and extract it to the /home/user/libtorch path.
>
> Example of `model_file`: "/home/user/model.prototxt". Examples of different types of model suffixes: TF: "model.pb" | CAFFE: "model.prototxt" | ONNX: "model.onnx" | MINDIR: "model.mindir" | TFLITE: "model.tflite" | PYTORCH: "model.pt or model.pth".
>
> `output_file` parameter descriptions: If `export_mindir` is set to `ModelType.MINDIR`, a MindSpore model will be generated, which uses `.mindir` as a suffix. If `export_mindir` is set to `ModelType.MINDIR_LITE`, a MindSpore Lite model will be generated, which uses `.ms` as a suffix. For example, input model is "/home/user/model.prototxt", and export_mindir uses default value, it will generate model named model.prototxt.ms in /home/user/ path.
>
> Caffe models are generally divided into two files: `*.prototxt` is the model structure, corresponding to the `model_file` parameter, and `model.caffemodel` is the model weights, corresponding to the `weight_file` parameter.
>
> The `config_file` configuration file uses `key = value` to define the relevant parameters. The quantization-related configuration parameters are detailed in [quantization after training](https://www.mindspore.cn/lite/docs/en/r2.0.0-alpha/use/post_training_quantization.html). The configuration parameters related to the extended functions are detailed in [Extended Configuration](https://www.mindspore.cn/lite/docs/en/r2.0.0-alpha/use/nnie.html#extension-configuration).
>
> The priority of `weight_fp16` is very low, for example if quantization is turned on, `weight_fp16` will not take effect again for weights that have already been quantized. In summary, this parameter will only take effect on serialization for the weights of Float32 in the model.
>
> `input_shape` is a parameter that the user may need to set in the following scenarios:
>
> - Usage 1: The input of the model to be transformed is dynamic shape, and the fixed-shape inference is to be used, then set this parameter to fixed-shape. After setting, when inference about the model after the Converter, the default input shape is the same as this parameter setting, and no resize operation is needed.
> - Usage 2: Regardless of whether the original input of the model to be transformed is a dynamic shape or not, use fixed-shape inference and make the performance of the model to be optimized as much as possible, set this parameter to fixed-shape. After setting, the model structure will be further optimized, but the transformed model may lose the characteristics of the dynamic shape (some operators strongly related to the shape will be fused).
> - Usage 3: When using the Converter function to generate code for Micro inference execution, it is recommended to configure this parameter to reduce the probability of errors during deployment. When the model contains a Shape operator or the model input to be transformed is a dynamic shape, this parameter must be configured to set a fixed shape, to support the associated shape optimization and code generation.
>
> `input_format`: Generally in three-way hardware scenarios with integrated NCHW specifications (e.g., [Usage Description of the Integrated NNIE](https://www.mindspore.cn/lite/docs/en/r2.0.0-alpha/use/nnie.html#usage-description-of-the-integrated-nnie)), setting to NCHW will result in more significant performance improvement than setting to NHWC. In other scenarios, users can also set up on-demand.
>
> The encryption and decryption function is only effective when set to `MSLITE_ENABLE_MODEL_ENCRYPTION=on` at [compilation](https://www.mindspore.cn/lite/docs/en/r2.0.0-alpha/use/build.html), and is only supported on Linux x86 platform, where the key is a hexadecimal representation of the string, such as the key is defined as `b'0123456789ABCDEF'` corresponding to the hexadecimal representation of `30313233343536373839414243444546`, and Linux platform users can use the `xxd` tool to convert the byte representation of the key to hexadecimal expression.
> Note that the encryption and decryption algorithms were updated in version 1.7, resulting in the new version of the Python interface not supporting the conversion of models exported from MindSpore encryption in version 1.6 and earlier.

### Usage Examples

The following is a selection of common examples to illustrate the use of the conversion command.

- Take the Caffe model LeNet as an example.

  ```python
  import mindspore_lite as mslite
  converter = mslite.Converter(fmk_type=mslite.FmkType.CAFFE, model_file="lenet.prototxt", output_file="lenet", weight_file="lenet.caffemodel")
  converter.converter()
  ```

  In this example, because the Caffe model is used, two input files, model structure and model weights, are required. Together with the other two required parameters, fmk type and output path, it can be executed successfully.

  The result is shown as:

  ```text
  CONVERT RESULT SUCCESS:0
  ```

  This means that the Caffe model has been successfully transformed into a MindSpore Lite model, obtaining the new file `lenet.ms`.

- Take MindSpore, TensorFlow Lite, TensorFlow and ONNX models as examples and execute the conversion command.

    - MindSpore model `model.mindir`

      ```python
      import mindspore_lite as mslite
      converter = mslite.Converter(fmk_type=mslite.FmkType.MINDIR, model_file="model.mindir", output_file="model")
      converter.converter()
      ```

    - TensorFlow Lite model `model.tflite`

      ```python
      import mindspore_lite as mslite
      converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="model.tflite", output_file="model")
      converter.converter()
      ```

    - TensorFlow model `model.pb`

      ```python
      import mindspore_lite as mslite
      converter = mslite.Converter(fmk_type=mslite.FmkType.TF, model_file="model.pb", output_file="model")
      converter.converter()
      ```

    - ONNX model `model.onnx`

      ```python
      import mindspore_lite as mslite
      converter = mslite.Converter(fmk_type=mslite.FmkType.ONNX, model_file="model.onnx", output_file="model")
      converter.converter()
      ```

    - PyTorch model `model.pt`

      ```python
      import mindspore_lite as mslite
      converter = mslite.Converter(fmk_type=mslite.FmkType.PYTORCH, model_file="model.pt", output_file="model")
      converter.converter()
      ```

    - PyTorch model `model.pth`

      ```python
      import mindspore_lite as mslite
      converter = mslite.Converter(fmk_type=mslite.FmkType.PYTORCH, model_file="model.pth", output_file="model")
      converter.converter()
      ```

     > Converting the PyTorch model has the following prerequisites: before compiling, export MSLITE_ENABLE_CONVERT_PYTORCH_MODEL=on is needed, and add libtorch environment variable: export LD_LIBRARY_PATH="/home/user/libtorch/lib:${LD_LIBRARY_PATH}" && export LIB_TORCH_PATH="/home/user/libtorch" before conversion. Users can download the [CPU version libtorch](https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.12.1%2Bcpu.zip) and extract it to the /home/user/libtorch path.

  In all of the above cases, the following conversion success message is displayed and the `model.ms` target file is obtained at the same time.

  ```text
  CONVERTER RESULT SUCCESS:0
  ```  