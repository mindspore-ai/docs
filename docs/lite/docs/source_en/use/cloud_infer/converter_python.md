# Using Python Interface to Perform Model Conversions

<a href="https://gitee.com/mindspore/docs/blob/master/docs/lite/docs/source_en/use/cloud_infer/converter_python.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## Overview

MindSpore Lite cloud-side inference supports model conversion via Python interface, supporting multiple types of model conversion, and the converted mindir models can be used for inference. The interface contains a variety of personalized parameters to provide a convenient conversion path for users. This tutorial describes how to use the [Python interface](https://www.mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.Converter.html) for model conversion.

The currently supported input model formats are MindSpore, TensorFlow Lite, Caffe, TensorFlow, and ONNX.

When the input model type is MindSpore, since it is already a `mindir` model, two approaches are recommended:

1. Inference is performed directly without offline conversion.

2. Set --NoFusion to false. The relevant optimization is done in the offline phase to reduce the initialization time of inference execution.

## Linux Environment Usage Instructions

### Environment Preparation

The following environment preparation is required for model conversion by using MindSpore Lite Python interface for cloud-side inference.

- [Compile](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/build.html) or [download](https://www.mindspore.cn/lite/docs/en/master/use/downloads.html) whl installation package for MindSpore Lite cloud-side inference with Converter component.

    > Currently, the installation package corresponding to Python 3.7 is available for download. If you need other Python versions, please use the compile function to generate the installation package.

- Then use the `pip install` command to install. After installation, you can use the following command to check if the installation is successful. If no error is reported, the installation is successful.

    ```bash
    python -c "import mindspore_lite"
    ```

### Directory Structure

After successful installation, you can use the `pip show mindspore_lite` command to see where the Python module for MindSpore Lite cloud-side inference is installed.

```text
mindspore_lite
├── __pycache__
├── include
├── lib
│   ├── _c_lite_wrapper.cpython-37m-x86_64-linux-gnu.so         # MindSpore Lite cloud-side inference python module encapsulates the dynamic library of the C++ interface framework
│   ├── libmindspore_converter.so                               # Dynamic library for model conversion
│   ├── libmindspore_core.so                                    # MindSpore Core Dynamic Library
│   ├── libmindspore_glog.so.0                                  # Glog dynamic library
│   ├── libmindspore-lite.so                                    # MindSpore Lite Dynamic Library for Cloud-Side inference
│   ├── libmslite_converter_plugin.so                           # Model Conversion Plugin
│   ├── libascend_pass_plugin.so                                # Register for Ascend Backend Graph Optimization Plugin Dynamic Library
│   ├── libmindspore_shared_lib.so                              # Adaptation of the dynamic library in the backend of Ascend
│   ├── libascend_kernel_plugin.so                              # Ascend backend kernel plugin
│   ├── libtensorrt_plugin.so                                   # tensorrt backend kernel plugin
│   ├── libopencv_core.so.4.5                                   # Dynamic library for OpenCV
│   ├── libopencv_imgcodecs.so.4.5                              # Dynamic library for OpenCV
│   └── libopencv_imgproc.so.4.5                                # Dynamic library for OpenCV
├── __init__.py        # Initialization package
├── context.py         # Code related to context interface
├── converter.py       # Code related to converter interface, conversion portal
├── model.py           # Code related to model, inference portal
├── tensor.py          # Code related to tensor interface
└── version.py         # MindSpore Lite cloud-side inference version number
```

### Description of Parameters

MindSpore Lite cloud-side inference model converter provides various parameter settings that users can choose to use according to their needs.

Usage scenario: Convert third-party models to generate MindSpore models.

Detailed descriptions of the parameters and their correspondence to the parameters in [Offline Conversion of Inference Models](https://www.mindspore.cn/lite/docs/en/master/use/cloud_infer/converter_tool.html) are provided below.

| Python interface model conversion parameters | Tpyes of parameters  | Parameters corresponding to the offline conversion of the model  |  Required or not   |  Description of parameters  | Value range | Default values | Remarks |
| -------- | ----- | -------- | ------- | ----- | --- | ---- | ---- |
| fmk_type | FmkType | `--fmk=<FMK>`  | Required | Input model frame type. | FmkType.TF, FmkType.CAFFE, FmkType.ONNX, FmkType.TFLITE | - |
| model_file | str | `--modelFile=<MODELFILE>` | Required | The path of the input model file for the conversion. | - | - |
| output_file | str | `--outputFile=<OUTPUTFILE>` | Required | The path to the output model at the time of conversion can be automatically generated with the `.mindir` suffix. | - | - |
| weight_file | str | `--weightFile=<WEIGHTFILE>` | Required when converting Caffe models | The path to the input model weights file. | - | "" |
| config_file | str | `--configFile=<CONFIGFILE>` | Not | Converter profile path to configure training post-quantization or offline splitting operators parallel or to disable the operator fusion function and set the plug-in to the so path, etc. | - | "" |
| input_shape | dict{string:list\[int]} | `--inputShape=<INPUTSHAPE>` | Not | Set the dimensions of the model input, and the order of the input dimensions is kept the same as the original model. For example: {"inTensor1": \[1, 32, 32, 32], "inTensor2": \[1, 1, 32, 32]} | - | None, None is equivalent to {} |
| export_mindir | ModelType | `--exportMindIR=<EXPORTMINDIR>` | Required | Set the type of the exported model file.| ModelType.MINDIR, ModelType.MINDIR_LITE | ModelType.MINDIR | The MINDIR model uses the MindSpore Lite cloud-side inference installation package, and MINDIR_LITE uses the MindSpore Lite device-side inference installation package|
| no_fusion | bool | `--NoFusion=<NOFUSION>` | Not | Whether to avoid fusion optimization, the default allows fusion optimization. | True, False | False |
| weight_fp16 | bool | `--fp16=<FP16>` | Not | Set whether the weights in Float32 data format need to be stored in Float16 data format during model serialization. | True, False | False | Not supported at the moment |
| input_format | Format | `--inputDataFormat=<INPUTDATAFORMAT>` | Not | Set the input format of the exported model, valid only for 4-dimensional inputs. | Format.NCHW, Format.NHWC | Format.NHWC | Not supported at the moment |
| input_data_type | DataType | `--inputDataType=<INPUTDATATYPE>` | Not | Set the data type of the quantized model input Tensor. Only valid if the quantization parameters (`scale` and `zero point`) of the model input Tensor are available. The default is to keep the same data type as the original model input Tensor. | DataType.FLOAT32, DataType.INT8, DataType.UINT8, DataType.UNKNOWN | DataType.FLOAT32 | Not supported at the moment |
| output_data_type | DataType | `--outputDataType=<OUTPUTDATATYPE>` | Not | Set the data type of the quantized model output Tensor. Only valid if the quantization parameters (`scale` and `zero point`) of the model output Tensor are available. The default is to keep the same data type as the original model output Tensor.   | DataType.FLOAT32、DataType.INT8、DataType.UINT8、DataType.UNKNOWN | DataType.FLOAT32 | Not supported at the moment |
| decrypt_key | str | `--decryptKey=<DECRYPTKEY>` | Not | Set the key used to load the cipher text MindIR. The key is expressed in hexadecimal and is only valid when `fmk_type` is MINDIR. | - | "" | Not supported at the moment |
| decrypt_mode | str | `--decryptMode=<DECRYPTMODE>` | Not | Set the mode to load cipher MindIR, only valid when `decrypt_key` is specified. | "AES-GCM", "AES-CBC" | "AES-GCM" | Not supported at the moment |
| enable_encryption | bool | `--encryption=<ENCRYPTION>` | Not | Whether to encrypt the model when exporting. Exporting encryption protects model integrity, but increases runtime initialization time. | True, False | False | Not supported at the moment |
| encrypt_key | str | `--encryptKey=<ENCRYPTKEY>` | Not | Set the key used to encrypt the file, in hexadecimal characters. Only supported when `decrypt_mode` is "AES-GCM" and the key length is 16. | - | "" | Not supported at the moment |
| infer | bool | `--infer=<INFER>` | Not | Whether to perform pre-inference at the completion of the conversion. | True, False | False | Not supported at the moment |
| train_model | bool | `--trainModel=<TRAINMODEL>` | Not | Whether the model will be trained on the device. | True, False | False | Not supported at the moment |

> For more information about the `fmk_type` parameter, see [FmkType](https://mindspore.cn/lite/api/en/master/mindspore_lite/mindspore_lite.FmkType.html)
>
> Example of `model_file`: "/home/user/model.prototxt". Examples of different types of model suffixes: TF: "model.pb" | CAFFE: "model.prototxt" | ONNX: "model.onnx" | TFLITE: "model.tflite".
>
> Caffe models are generally divided into two files: `*.prototxt` is model structure, corresponding to the `model_file` parameter, and `model.caffemodel` is model weights, corresponding to the `--weightFile` parameter.
>
> The `config_file` configuration file uses the `key = value` approach to define the relevant parameters.
>
> `input_shape` is a parameter that the user may need to set in the following scenarios:
>
> - Usage 1: The input of the model to be converted is dynamic shape, and the fixed-shape inference is prepared, then this parameter is set to fixed-shape. After setting, when inference about the model after the Converter, the default input shape is the same as this parameter setting, and no resize operation is needed.
> - Usage 2: Regardless of whether the original input of the model to be converted is a dynamic shape, use fixed-shape inference and want the performance of the model to be optimized as much as possible, then set this parameter to fixed-shape. After setting, the model structure will be further optimized, but the converted model may lose the characteristics of the dynamic shape (some operators strongly related to the shape will be fused).
>

### Usage Examples

The following selects common examples to illustrate the use of the conversion command.

- Take the Caffe model LeNet as an example

    ```python
    import mindspore_lite as mslite
    converter = mslite.Converter(fmk_type=mslite.FmkType.CAFFE, model_file="lenet.prototxt",output_file="lenet", weight_file="lenet.caffemodel",export_mindir=mslite.ModelType.MINDIR)
    converter.converter()
    ```

    In this example, because the Caffe model is used, two input files, model structure and model weights, are required. Together with the other two required parameters, fmk type and output path, it can be executed successfully.

    The result is shown as:

    ```text
    CONVERT RESULT SUCCESS:0
    ```

    This indicates that the Caffe model has been successfully converted into a MindSpore Lite cloud-side inference model, obtaining the new file `lenet.mindir`.

- Take MindSpore, TensorFlow Lite, TensorFlow and ONNX models as examples and execute the conversion command.

    - MindSpore model `model.mindir`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter(fmk_type=mslite.FmkType.MINDIR, model_file="model.mindir",output_file="model",export_mindir=mslite.ModelType.MINDIR, no_fusion=False)
        converter.converter()
        ```

    - TensorFlow Lite model `model.tflite`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter(fmk_type=mslite.FmkType.TFLITE, model_file="model.tflite",output_file="model",export_mindir=mslite.ModelType.MINDIR)
        converter.converter()
        ```

    - TensorFlow model `model.pb`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter(fmk_type=mslite.FmkType.TF, model_file="model.pb", output_file="model",export_mindir=mslite.ModelType.MINDIR)
        converter.converter()
        ```

    - ONNX model `model.onnx`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter(fmk_type=mslite.FmkType.ONNX, model_file="model.onnx", output_file="model",export_mindir=mslite.ModelType.MINDIR)
        converter.converter()
        ```

    In all of the above cases, the following conversion success message is displayed and the `model.mindir` target file is obtained at the same time.

    ```text
    CONVERTER RESULT SUCCESS:0
    ```
