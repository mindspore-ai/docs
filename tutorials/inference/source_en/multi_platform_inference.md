# Multi-Platform Inference

`Linux` `Ascend` `GPU` `CPU` `Inference Application` `Beginner` `Intermediate` `Expert`

<!-- TOC -->

- [Multi-Platform Inference](#multi-platform-inference)
    - [Overview](#overview)
    - [Inference on the Ascend 910 AI processor](#inference-on-the-ascend-910-ai-processor)
        - [Inference Using a Checkpoint File](#inference-using-a-checkpoint-file)
    - [Inference on the Ascend 310 AI processor](#inference-on-the-ascend-310-ai-processor)
        - [Inference Using an ONNX or AIR File](#inference-using-an-onnx-or-air-file)
    - [Inference on a GPU](#inference-on-a-gpu)
        - [Inference Using a Checkpoint File](#inference-using-a-checkpoint-file-1)
        - [Inference Using an ONNX File](#inference-using-an-onnx-file)
    - [Inference on a CPU](#inference-on-a-cpu)
        - [Inference Using a Checkpoint File](#inference-using-a-checkpoint-file-2)
        - [Inference Using an ONNX File](#inference-using-an-onnx-file-1)
    - [On-Device Inference](#on-device-inference)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.0/tutorials/inference/source_en/multi_platform_inference.md" target="_blank"><img src="../_static/logo_source.png"></a>

## Overview

Models trained by MindSpore support the inference on different hardware platforms. This document describes the inference process on each platform.

The inference can be performed in either of the following methods based on different principles:
- Use a checkpoint file for inference. That is, use the inference API to load data and the checkpoint file for inference in the MindSpore training environment.
- Convert the checkpiont file into a common model format, such as ONNX or AIR, for inference. The inference environment does not depend on MindSpore. In this way, inference can be performed across hardware platforms as long as the platform supports ONNX or AIR inference. For example, models trained on the Ascend 910 AI processor can be inferred on the GPU or CPU.

MindSpore supports the following inference scenarios based on the hardware platform:

| Hardware Platform       | Model File Format | Description                              |
| ----------------------- | ----------------- | ---------------------------------------- |
| Ascend 910 AI processor | Checkpoint        | The training environment dependency is the same as that of MindSpore. |
| Ascend 310 AI processor | ONNX or AIR       | Equipped with the ACL framework and supports the model in OM format. You need to use a tool to convert a model into the OM format. |
| GPU                     | Checkpoint        | The training environment dependency is the same as that of MindSpore. |
| GPU                     | ONNX              | Supports ONNX Runtime or SDK, for example, TensorRT. |
| CPU                     | Checkpoint        | The training environment dependency is the same as that of MindSpore. |
| CPU                     | ONNX              | Supports ONNX Runtime or SDK, for example, TensorRT. |

> Open Neural Network Exchange (ONNX) is an open file format designed for machine learning. It is used to store trained models. It enables different AI frameworks (such as PyTorch and MXNet) to store model data in the same format and interact with each other. For details, visit the ONNX official website <https://onnx.ai/>.

> Ascend Intermediate Representation (AIR) is an open file format defined by Huawei for machine learning and can better adapt to the Ascend AI processor. It is similar to ONNX.

> Ascend Computer Language (ACL) provides C++ API libraries for users to develop deep neural network applications, including device management, context management, stream management, memory management, model loading and execution, operator loading and execution, and media data processing. It matches the Ascend AI processor and enables hardware running management and resource management.

> Offline Model (OM) is supported by the Huawei Ascend AI processor. It implements preprocessing functions that can be completed without devices, such as operator scheduling optimization, weight data rearrangement and compression, and memory usage optimization.

> NVIDIA TensorRT is an SDK for high-performance deep learning inference. It includes a deep learning inference optimizer and runtime to improve the inference speed of the deep learning model on edge devices. For details, see <https://developer.nvidia.com/tensorrt>.

## Inference on the Ascend 910 AI processor

### Inference Using a Checkpoint File

1. Use the `model.eval` interface for model validation. 

   1.1 Local Storage

     When the pre-trained models are saved in local, the steps of performing inference on validation dataset are as follows: firstly creating a model, then loading model and parameters using `load_checkpoint` and `load_param_into_net` in `mindspore.train.serialization` module, and finally performing inference on validation dataset once created. The processing method of the validation dataset is the same as that of the training dataset.

    ```python
    network = LeNet5(cfg.num_classes)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(network, param_dict)
    dataset = create_dataset(os.path.join(args.data_path, "test"),
                             cfg.batch_size,
                             1)
    acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
    ```
    In the preceding information:  
    `model.eval` is an API for model validation. For details about the API, see <https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.html#mindspore.Model.eval>.
    > Inference sample code: <https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/lenet/eval.py>.

    1.2 Remote Storage
    
    When the pre-trained models are saved remotely, the steps of performing inference on validation dataset are as follows: firstly determine which model to be used, then loading model and parameters using `mindspore_hub.load`, and finally performing inference on validation dataset once created. The processing method of the validation dataset is the same as that of the training dataset.

    ```python
    model_uid = "mindspore/ascend/0.7/googlenet_v1_cifar10"  # using GoogleNet as an example.
    network = mindspore_hub.load(model_uid, num_classes=10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), cfg.lr, cfg.momentum)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    print("============== Starting Testing ==============")
    dataset = create_dataset(os.path.join(args.data_path, "test"),
                             cfg.batch_size,
                             1)
    acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
    print("============== {} ==============".format(acc))
    ``` 
    In the preceding information:
        
    `mindpsore_hub.load` is an API for loading model parameters. PLease check the details in <https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore_hub/mindspore_hub.html#module-mindspore_hub>.

2. Use the `model.predict` API to perform inference.
   ```python
   model.predict(input_data)
   ```
   In the preceding information:  
   `model.predict` is an API for inference. For details about the API, see <https://www.mindspore.cn/doc/api_python/en/r1.0/mindspore/mindspore.html#mindspore.Model.predict>.

## Inference on the Ascend 310 AI processor

### Inference Using an ONNX or AIR File

The Ascend 310 AI processor is equipped with the ACL framework and supports the OM format which needs to be converted from the model in ONNX or AIR format. For inference on the Ascend 310 AI processor, perform the following steps:

1. Generate a model in ONNX or AIR format on the training platform. For details, see [Export AIR Model](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_and_load_model.html#export-air-model) and [Export ONNX Model](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_and_load_model.html#export-onnx-model).

2. Convert the ONNX or AIR model file into an OM model file and perform inference.
   - For performing inference in the cloud environment (ModelArt), see the [Ascend 910 training and Ascend 310 inference samples](https://support.huaweicloud.com/en-us/bestpractice-modelarts/modelarts_10_0026.html).
   - For details about the local bare-metal environment where the Ascend 310 AI processor is deployed in local (compared with the cloud environment), see the document of the Ascend 310 AI processor software package.

## Inference on a GPU

### Inference Using a Checkpoint File

The inference is the same as that on the Ascend 910 AI processor.

### Inference Using an ONNX File

1. Generate a model in ONNX format on the training platform. For details, see [Export ONNX Model](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_and_load_model.html#export-onnx-model).

2. Perform inference on a GPU by referring to the runtime or SDK document. For example, use TensorRT to perform inference on the NVIDIA GPU. For details, see [TensorRT backend for ONNX](https://github.com/onnx/onnx-tensorrt).

## Inference on a CPU

### Inference Using a Checkpoint File
The inference is the same as that on the Ascend 910 AI processor.

### Inference Using an ONNX File
Similar to the inference on a GPU, the following steps are required:

1. Generate a model in ONNX format on the training platform. For details, see [Export ONNX Model](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_and_load_model.html#export-onnx-model).

2. Perform inference on a CPU by referring to the runtime or SDK document. For details about how to use the ONNX Runtime, see the [ONNX Runtime document](https://github.com/microsoft/onnxruntime).

## On-Device Inference

MindSpore Lite is an inference engine for on-device inference. For details, see [Export MINDIR Model](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_and_load_model.html#export-mindir-model) and [On-Device Inference](https://www.mindspore.cn/lite/en).
