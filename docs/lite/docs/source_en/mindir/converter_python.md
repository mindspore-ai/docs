# Using Python Interface to Perform Model Conversions

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/mindir/converter_python.md)

## Overview

MindSpore Lite cloud-side inference supports model conversion via Python interface, supporting multiple types of model conversion, and the converted mindir models can be used for inference. The interface contains a variety of personalized parameters to provide a convenient conversion path for users. This tutorial describes how to use the [Python interface](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Converter.html) for model conversion.

The currently supported input model formats are MindSpore, TensorFlow Lite, Caffe, TensorFlow, and ONNX.

When the input model type is MindSpore, since it is already a `mindir` model, two approaches are recommended:

1. Inference is performed directly without offline conversion.

2. When using offline conversion, setting `optimize` to `general` in CPU/GPU hardware backend (for general optimization), setting `optimize` to `gpu_oriented` in GPU hardware (for GPU extra optimization based on general optimization), setting `optimize` to `ascend_oriented` in Ascend hardware. The relevant optimization is done in the offline phase to reduce the initialization time of inference execution.

## Linux Environment Usage Instructions

### Environment Preparation

The following environment preparation is required for model conversion by using MindSpore Lite Python interface for cloud-side inference.

- [Compile](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/build.html) or [download](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/use/downloads.html) whl installation package for MindSpore Lite cloud-side inference with Converter component.

    > Currently, the installation package corresponding to Python 3.7 is available for download. If you need other Python versions, please use the compile function to generate the installation package.

- Then use the `pip install` command to install. After installation, you can use the following command to check if the installation is successful. If no error is reported, the installation is successful.

    ```bash
    python -c "import mindspore_lite"
    ```

- After installation, you can use the following command to check if MindSpore Lite built-in AKG is successfully installed. If no error is reported, the installation is successful.

  ```bash
  python -c "import mindspore_lite.akg"
  ```

### Directory Structure

After successful installation, you can use the `pip show mindspore_lite` command to see where the Python module for MindSpore Lite cloud-side inference is installed.

```text
mindspore_lite
├── __pycache__
├── akg                                                         # AKG-related interfaces
├── include
├── lib
|   ├── libakg.so                                               # Dynamic link libraries used by AKG
│   ├── _c_lite_wrapper.cpython-37m-x86_64-linux-gnu.so         # MindSpore Lite cloud-side inference python module encapsulates the dynamic library of the C++ interface framework
│   ├── libmindspore_converter.so                               # Dynamic library for model conversion
│   ├── libmindspore_core.so                                    # MindSpore Core Dynamic Library
│   ├── libmindspore_glog.so.0                                  # Glog dynamic library
│   ├── libmindspore-lite.so                                    # MindSpore Lite Dynamic Library for Cloud-Side inference
│   ├── libmslite_converter_plugin.so                           # Model Conversion Plugin
│   ├── libascend_pass_plugin.so                                # Register for Ascend Backend Graph Optimization Plugin Dynamic Library
│   ├── libmslite_shared_lib.so                                 # Adaptation of the dynamic library in the backend of Ascend
│   ├── libascend_kernel_plugin.so                              # Ascend backend kernel plugin
│   ├── libtensorrt_plugin.so                                   # tensorrt backend kernel plugin
│   ├── libopencv_core.so.4.5                                   # Dynamic library for OpenCV
│   ├── libopencv_imgcodecs.so.4.5                              # Dynamic library for OpenCV
│   └── libopencv_imgproc.so.4.5                                # Dynamic library for OpenCV
├── __init__.py        # Initialization package
├── _checkparam.py     # Check parameter tool
├── context.py         # Code related to context interface
├── converter.py       # Code related to converter interface, conversion portal
├── model.py           # Code related to model, inference portal
├── tensor.py          # Code related to tensor interface
└── version.py         # MindSpore Lite cloud-side inference version number
```

### Description of Attributes

MindSpore Lite cloud-side inference model converter provides various attribute settings that users can choose to use according to their needs.

Detailed descriptions of the parameters and their correspondence to the parameters in [Offline Conversion of Inference Models](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool.html) are provided below.

| Converter attributes | Types of attributes  | Parameters corresponding to the offline conversion of the model  | Description  | Value range | Remarks |
| -------- | ----- | -------- | ------- | ---- | ---- |
| decrypt_key | str | `--decryptKey=<DECRYPTKEY>` | Set the key used to load the cipher text MindIR. The key is expressed in hexadecimal and is only valid when `fmk_type` is MINDIR. | - | - |
| decrypt_mode | str | `--decryptMode=<DECRYPTMODE>` | Set the mode to load cipher MindIR, only valid when `decrypt_key` is specified. | "AES-GCM", "AES-CBC" | - |
| device | str | `--device=<DEVICE>` | Set target device when converter model. The use case is when on the Ascend device, if you need to the converted model to have the ability to use Ascend backend to perform inference, you can set the attribute. If it is not set, the converted model will use CPU backend to perform inference by default. | "Ascend" | - |
| encrypt_key | str | `--encryptKey=<ENCRYPTKEY>` | Set the key used to encrypt the file, in hexadecimal characters. Only supported when `decrypt_mode` is "AES-GCM" and the key length is 16. | - | - |
| enable_encryption | bool | `--encryption=<ENCRYPTION>` | Whether to encrypt the model when exporting. Exporting encryption protects model integrity, but increases runtime initialization time. | True, False | - |
| infer | bool | `--infer=<INFER>` | Whether to perform pre-inference at the completion of the conversion. | True, False | - |
| input_data_type | DataType | `--inputDataType=<INPUTDATATYPE>` | Set the data type of the quantized model input Tensor. Only valid if the quantization parameters (`scale` and `zero point`) of the model input Tensor are available. The default is to keep the same data type as the original model input Tensor. | DataType.FLOAT32, DataType.INT8, DataType.UINT8, DataType.UNKNOWN | - |
| input_format | Format | `--inputDataFormat=<INPUTDATAFORMAT>` | Set the input format of the exported model, valid only for 4-dimensional inputs. | Format.NCHW, Format.NHWC | - |
| input_shape | dict{string:list\[int]} | `--inputShape=<INPUTSHAPE>` | Set the dimensions of the model input, and the order of the input dimensions is kept the same as the original model. For example: {"inTensor1": \[1, 32, 32, 32], "inTensor2": \[1, 1, 32, 32]} | - |
| optimize | str | `--optimize=<OPTIMIZE>` | Set the mode of optimization during the offline conversion. | "none", "general", "gpu_oriented", "ascend_oriented" | - |
| output_data_type | DataType | `--outputDataType=<OUTPUTDATATYPE>` | Set the data type of the quantized model output Tensor. Only valid if the quantization parameters (`scale` and `zero point`) of the model output Tensor are available. The default is to keep the same data type as the original model output Tensor. | DataType.FLOAT32, DataType.INT8, DataType.UINT8, DataType.UNKNOWN | - |
| save_type | ModelType | `--saveType=<SAVETYPE>` | Required | Set the model type needs to be export. | ModelType.MINDIR | The MINDIR model uses the MindSpore Lite cloud-side inference installation package |
| weight_fp16 | bool | `--fp16=<FP16>` | Set whether the weights in float32 data format need to be stored in float16 data format during model serialization. | True, False | - |

> - The encryption and decryption function only takes effect when `MSLITE_ENABLE_MODEL_ENCRYPTION=on` is set at [compile](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/build.html) time and only supports Linux x86 platforms. `decrypt_key` and `encrypt_key` are string expressed in hexadecimal. For example, if encrypt_key is set as "30313233343637383939414243444546", the corresponding hexadecimal expression is '(b)0123456789ABCDEF' . Linux platform users can use the' xxd 'tool to convert the key expressed in bytes into hexadecimal expressions.
>
> - `input_shape` is a attribute that the user may need to set in the following scenarios:
>
>   - Usage 1: The input of the model to be converted is dynamic shape, and the fixed-shape inference is prepared, then this attribute is set to fixed-shape. After setting, when inference about the model after the Converter, the default input shape is the same as this attribute setting, and no resize operation is needed.
>   - Usage 2: Regardless of whether the original input of the model to be converted is a dynamic shape, use fixed-shape inference and want the performance of the model to be optimized as much as possible, then set this attribute to fixed-shape. After setting, the model structure will be further optimized, but the converted model may lose the characteristics of the dynamic shape (some operators strongly related to the shape will be fused).
>
> - `optimize` is an attribute, it used to set the mode of optimization during the offline conversion.
>
>   - If this attribute is set to "none", no relevant graph optimization operations will be performed during the offline conversion phase of the model, and the relevant graph optimization operations will be performed during the execution of the inference phase. The advantage of this attribute is that the converted model can be deployed directly to any CPU/GPU/Ascend hardware backend since it is not optimized in a specific way, while the disadvantage is that the initialization time of the model increases during inference execution.
>   - If this attribute is set to "general", general optimization will be performed, such as constant folding and operator fusion (the converted model only supports CPU/GPU hardware backend, not Ascend backend).
>   - If this parameter is set to "gpu_oriented", the general optimization and extra optimization for GPU hardware will be performed (the converted model only supports GPU hardware backend).
>   - If this attribute is set to "ascend_oriented", the optimization for Ascend hardware will be performed (the converted model only supports Ascend hardware backend).
>

### Method of convert

Usage scenario: Convert a third-party model into a MindSpore model. You can call the convert method multiple times to convert multiple models.

Detailed descriptions of the parameters and their correspondence to the parameters in [Offline Conversion of Inference Models](https://www.mindspore.cn/lite/docs/en/r2.6.0rc1/mindir/converter_tool.html) are provided below.

| Method of convert parameters | Tpyes of parameters  | Parameters corresponding to the offline conversion of the model  |  Required or not   |  Description of parameters  | Value range | Default values |
| -------- | ----- | -------- | ------- | ----- | --- | ---- |
| fmk_type | FmkType | `--fmk=<FMK>`  | Required | Input model frame type. | FmkType.TF, FmkType.CAFFE, FmkType.ONNX, FmkType.TFLITE | - |
| model_file | str | `--modelFile=<MODELFILE>` | Required | The path of the input model file for the conversion. | - | - |
| output_file | str | `--outputFile=<OUTPUTFILE>` | Required | The path to the output model at the time of conversion can be automatically generated with the `.mindir` suffix. | - | - |
| weight_file | str | `--weightFile=<WEIGHTFILE>` | Required when converting Caffe models | The path to the input model weights file. | - | "" |
| config_file | str | `--configFile=<CONFIGFILE>` | Not | Converter profile path to configure training post-quantization or offline splitting operators parallel or to disable the operator fusion function and set the plug-in to the so path, etc. | - | "" |

> For more information about the `fmk_type` parameter, see [FmkType](https://mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.FmkType.html)
>
> Example of `model_file`: "/home/user/model.prototxt". Examples of different types of model suffixes: TF: "model.pb" | CAFFE: "model.prototxt" | ONNX: "model.onnx" | TFLITE: "model.tflite".
>
> Caffe models are generally divided into two files: `*.prototxt` is model structure, corresponding to the `model_file` parameter, and `model.caffemodel` is model weights, corresponding to the `--weightFile` parameter.
>
> The `config_file` configuration file uses the `key = value` approach to define the relevant parameters.
>

### Usage Examples

The following selects common examples to illustrate the use of the conversion command.

- Take the Caffe model LeNet as an example

    ```python
    import mindspore_lite as mslite
    converter = mslite.Converter()
    converter.save_type = mslite.ModelType.MINDIR
    converter.optimize = "none"
    converter.convert(fmk_type=mslite.FmkType.CAFFE, model_file="lenet.prototxt",output_file="lenet", weight_file="lenet.caffemodel")
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
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = "general"
        converter.convert(fmk_type=mslite.FmkType.MINDIR, model_file="model.mindir",output_file="model")
        ```

    - TensorFlow Lite model `model.tflite`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = "none"
        converter.convert(fmk_type=mslite.FmkType.TFLITE, model_file="model.tflite",output_file="model")
        ```

    - TensorFlow model `model.pb`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = "none"
        converter.convert(fmk_type=mslite.FmkType.TF, model_file="model.pb", output_file="model")
        ```

    - ONNX model `model.onnx`

        ```python
        import mindspore_lite as mslite
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = "none"
        converter.convert(fmk_type=mslite.FmkType.ONNX, model_file="model.onnx", output_file="model")
        ```

    In all of the above cases, the following conversion success message is displayed and the `model.mindir` target file is obtained at the same time.

    ```text
    CONVERT RESULT SUCCESS:0
    ```

### Advanced Usage

#### Online conversion

get_config_info method and set_config_info method is used for online conversion. Please refer to the [set_config_info](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/mindspore_lite/mindspore_lite.Converter.html#mindspore_lite.Converter.set_config_info) for details.
