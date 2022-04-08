# Inference Model Overview

`Ascend` `GPU` `CPU` `Inference Application`

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/model_infer/multi_platform_inference.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

MindSpore can execute inference tasks on different hardware platforms based on trained models.

## Model Files

MindSpore can save two types of data: training parameters and network models that contain parameter information.

- Training parameters are stored in the checkpoint format.
- Network models are stored in the MindIR, AIR, or ONNX format.

Basic concepts and application scenarios of these formats are as follows:

- Checkpoint
    - Checkpoint uses the Protocol Buffers format and stores all network parameter values.
    - It is generally used to resume training after a training task is interrupted or executes a fine-tune task after training.
- MindSpore IR (MindIR)
    - MindIR is a graph-based function-like IR of MindSpore and defines scalable graph structures and operator IRs.
    - It eliminates model differences between different backends and is generally used to perform inference tasks across hardware platforms.
- Open Neural Network Exchange (ONNX)
    - ONNX is an open format built to represent machine learning models.
    - It is generally used to transfer models between different frameworks or used on the inference engine ([TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/python_api/index.html)).
    - At present, mindspire only supports the export of ONNX model, and does not support loading onnx model for inference. Currently, the models supported for export are resnet50, yolov3_ darknet53, YOLOv4 and BERT. These models can be used on [ONNX Runtime](https://onnxruntime.ai/).
- Ascend Intermediate Representation (AIR)
    - AIR is an open file format defined by Huawei for machine learning.
    - It adapts to Huawei AI processors well and is generally used to execute inference tasks on Ascend 310.

## Inference Execution

Inference can be classified into the following two modes based on the application environment:

1. Local inference

    Load a checkpoint file generated during network training and call the `model.predict` API for inference and validation. For details, see [Online Inference with Checkpoint](https://www.mindspore.cn/tutorials/experts/en/master/model_infer/online_inference.html).

2. Cross-platform inference

    Use a network definition and a checkpoint file, call the `export` API to export a model file, and perform inference on different platforms. Currently, MindIR, ONNX, and AIR (on only Ascend AI Processors) models can be exported. For details, see [Saving Models](https://www.mindspore.cn/tutorials/experts/en/master/model_infer/save_model.html).

## Introduction to MindIR

MindSpore defines logical network structures and operator attributes through a unified IR, and decouples model files in MindIR format from hardware platforms to implement one-time training and multiple-time deployment.

1. Overview

    As a unified model file of MindSpore, MindIR stores network structures and weight parameter values. In addition, it can be deployed on the on-cloud Serving and the on-device Lite platforms to execute inference tasks.

    A MindIR file supports the deployment of multiple hardware forms.

    - On-cloud deployment and inference on Serving: After MindSpore trains and generates a MindIR model file, the file can be directly sent to MindSpore Serving for loading and inference. No additional model conversion is required. This ensures that models on different hardware such as Ascend, GPU, and CPU are unified.
    - On-device inference and deployment on Lite: MindIR can be directly used for Lite deployment. In addition, to meet the lightweight requirements on devices, the model miniaturization and conversion functions are provided. An original MindIR model file can be converted from the Protocol Buffers format to the FlatBuffers format for storage, and the network structure is lightweight to better meet the performance and memory requirements on devices.

2. Application Scenarios

    Use a network definition and a checkpoint file to export a MindIR model file, and then execute inference based on different requirements, for example, [Inference Using the MindIR Model on Ascend 310 AI Processors](https://www.mindspore.cn/tutorials/experts/en/master/model_infer/multi_platform_inference_ascend_310_mindir.html), [MindSpore Serving-based Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/master/serving_example.html), and [Inference on Devices](https://www.mindspore.cn/lite/docs/en/master/index.html).
