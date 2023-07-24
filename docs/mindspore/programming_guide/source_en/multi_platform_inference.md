# Inference Model Overview

`Linux` `Ascend` `GPU` `CPU` `Inference Application` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/programming_guide/source_en/multi_platform_inference.md)

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
    - It is generally used to transfer models between different frameworks or used on the inference engine ([TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)).
- Ascend Intermediate Representation (AIR)
    - AIR is an open file format defined by Huawei for machine learning.
    - It adapts to Huawei AI processors well and is generally used to execute inference tasks on Ascend 310.

## Inference Execution

Inference can be classified into the following two modes based on the application environment:

1. Local inference

    Load a checkpoint file generated during network training and call the `model.predict` API for inference and validation. For details, see [Inference Using a Checkpoint File with Single Device](https://www.mindspore.cn/docs/programming_guide/en/r1.3/multi_platform_inference_ascend_910.html#checkpoint).

2. Cross-platform inference

    Use a network definition and a checkpoint file, call the `export` API to export a model file, and perform inference on different platforms. Currently, MindIR, ONNX, and AIR (on only Ascend AI Processors) models can be exported. For details, see [Saving Models](https://www.mindspore.cn/docs/programming_guide/en/r1.3/save_model.html).

## Introduction to MindIR

MindSpore defines logical network structures and operator attributes through a unified IR, and decouples model files in MindIR format from hardware platforms to implement one-time training and multiple-time deployment.

1. Overview

    As a unified model file of MindSpore, MindIR stores network structures and weight parameter values. In addition, it can be deployed on the on-cloud Serving and the on-device Lite platforms to execute inference tasks.

    A MindIR file supports the deployment of multiple hardware forms.

    - On-cloud deployment and inference on Serving: After MindSpore trains and generates a MindIR model file, the file can be directly sent to MindSpore Serving for loading and inference. No additional model conversion is required. This ensures that models on different hardware such as Ascend, GPU, and CPU are unified.
    - On-device inference and deployment on Lite: MindIR can be directly used for Lite deployment. In addition, to meet the lightweight requirements on devices, the model miniaturization and conversion functions are provided. An original MindIR model file can be converted from the Protocol Buffers format to the FlatBuffers format for storage, and the network structure is lightweight to better meet the performance and memory requirements on devices.

2. Application Scenarios

    Use a network definition and a checkpoint file to export a MindIR model file, and then execute inference based on different requirements, for example, [Inference Using the MindIR Model on Ascend 310 AI Processors](https://www.mindspore.cn/docs/programming_guide/en/r1.3/multi_platform_inference_ascend_310_mindir.html), [MindSpore Serving-based Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/r1.3/serving_example.html), and [Inference on Devices](https://www.mindspore.cn/lite/docs/en/r1.3/index.html).

### Networks Supported by MindIR

<table class="docutils">
<tr>
  <td>AlexNet</td>
  <td>BERT</td>
  <td>BGCF</td>
</tr>
<tr>
  <td>CenterFace</td>
  <td>CNN&CTC</td>
  <td>DeepLabV3</td>
</tr>
<tr>
  <td>DenseNet121</td>
  <td>Faster R-CNN</td>
  <td>GAT</td>
</tr>
<tr>
  <td>GCN</td>
  <td>GoogLeNet</td>
  <td>LeNet</td>
</tr>
<tr>
  <td>Mask R-CNN</td>
  <td>MASS</td>
  <td>MobileNetV2</td>
</tr>
<tr>
  <td>NCF</td>
  <td>PSENet</td>
  <td>ResNet</td>
</tr>
<tr>
  <td>ResNeXt</td>
  <td>InceptionV3</td>
  <td>SqueezeNet</td>
</tr>
<tr>
  <td>SSD</td>
  <td>Transformer</td>
  <td>TinyBert</td>
</tr>
<tr>
  <td>UNet2D</td>
  <td>VGG16</td>
  <td>Wide&Deep</td>
</tr>
<tr>
  <td>YOLOv3</td>
  <td>YOLOv4</td>
  <td></td>
</tr>
</table>
