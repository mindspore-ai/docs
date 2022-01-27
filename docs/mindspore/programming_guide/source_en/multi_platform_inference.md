# Inference Model Overview

`Ascend` `GPU` `CPU` `Inference Application`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/programming_guide/source_en/multi_platform_inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

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
    - At present, mindspire only supports the export of ONNX model, and does not support loading onnx model for inference. Currently, the models supported for export are resnet50, yolov3_ darknet53, YOLOv4 and BERT. These models can be used on [ONNX Runtime](https://onnxruntime.ai/).
- Ascend Intermediate Representation (AIR)
    - AIR is an open file format defined by Huawei for machine learning.
    - It adapts to Huawei AI processors well and is generally used to execute inference tasks on Ascend 310.

## Inference Execution

Inference can be classified into the following two modes based on the application environment:

1. Local inference

    Load a checkpoint file generated during network training and call the `model.predict` API for inference and validation. For details, see [Online Inference with Checkpoint](https://www.mindspore.cn/docs/programming_guide/en/master/online_inference.html).

2. Cross-platform inference

    Use a network definition and a checkpoint file, call the `export` API to export a model file, and perform inference on different platforms. Currently, MindIR, ONNX, and AIR (on only Ascend AI Processors) models can be exported. For details, see [Saving Models](https://www.mindspore.cn/docs/programming_guide/en/master/save_model.html).

## Introduction to MindIR

MindSpore defines logical network structures and operator attributes through a unified IR, and decouples model files in MindIR format from hardware platforms to implement one-time training and multiple-time deployment.

1. Overview

    As a unified model file of MindSpore, MindIR stores network structures and weight parameter values. In addition, it can be deployed on the on-cloud Serving and the on-device Lite platforms to execute inference tasks.

    A MindIR file supports the deployment of multiple hardware forms.

    - On-cloud deployment and inference on Serving: After MindSpore trains and generates a MindIR model file, the file can be directly sent to MindSpore Serving for loading and inference. No additional model conversion is required. This ensures that models on different hardware such as Ascend, GPU, and CPU are unified.
    - On-device inference and deployment on Lite: MindIR can be directly used for Lite deployment. In addition, to meet the lightweight requirements on devices, the model miniaturization and conversion functions are provided. An original MindIR model file can be converted from the Protocol Buffers format to the FlatBuffers format for storage, and the network structure is lightweight to better meet the performance and memory requirements on devices.

2. Application Scenarios

    Use a network definition and a checkpoint file to export a MindIR model file, and then execute inference based on different requirements, for example, [Inference Using the MindIR Model on Ascend 310 AI Processors](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference_ascend_310_mindir.html), [MindSpore Serving-based Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/master/serving_example.html), and [Inference on Devices](https://www.mindspore.cn/lite/docs/en/master/index.html).

### Networks Supported by MindIR

<table class="docutils">
<tr>
  <td>AdvancedEast</td>
  <td>AlexNet</td>
  <td>AutoDis</td>
  <td>BERT</td>
  <td>BGCF</td>
  <td>CenterFace</td>
</tr>
<tr>
  <td>CNN</td>
  <td>CNN&CTC</td>
  <td>CRNN</td>
  <td>CSPDarkNet53</td>
  <td>CTPN</td>
  <td>DeepFM</td>
</tr>
<tr>
  <td>DeepLabV3</td>
  <td>DeepText</td>
  <td>DenseNet121</td>
  <td>DPN</td>
  <td>DS-CNN</td>
  <td>FaceAttribute</td>
</tr>
<tr>
  <td>FaceDetection</td>
  <td>FaceQualityAssessment</td>
  <td>FaceRecognition</td>
  <td>FaceRecognitionForTracking</td>
  <td>Faster R-CNN</td>
  <td>FasterRcnn-ResNet50</td>
</tr>
<tr>
  <td>FasterRcnn-ResNet101</td>
  <td>FasterRcnn-ResNet152</td>
  <td>FCN</td>
  <td>FCN-4</td>
  <td>GAT</td>
  <td>GCN</td>
</tr>
<tr>
  <td>GoogLeNet</td>
  <td>GRU</td>
  <td>hardnet</td>
  <td>InceptionV3</td>
  <td>InceptionV4</td>
  <td>LeNet</td>
</tr>
<tr>
  <td>LSTM-SegtimentNet</td>
  <td>Mask R-CNN</td>
  <td>MaskRCNN_MobileNetV1</td>
  <td>MASS</td>
  <td>MobileNetV1</td>
  <td>MobileNetV2</td>
</tr>
<tr>
  <td>NCF</td>
  <td>PSENet</td>
  <td>ResNet18</td>
  <td>ResNet50</td>
  <td>ResNet101</td>
  <td>ResNet152</td>
</tr>
<tr>
  <td>ResNetV2-50</td>
  <td>ResNetV2-101</td>
  <td>ResNetV2-152</td>
  <td>SE-Net</td>
  <td>SSD-MobileNetV2</td>
  <td>ResNext50</td>
</tr>
<tr>
  <td>ResNext101</td>
  <td>RetinaNet</td>
  <td>Seq2Seq(Attention)</td>
  <td>SE-ResNet50</td>
  <td>ShuffleNetV1</td>
  <td>SimplePoseNet</td>
</tr>
<tr>
  <td>SqueezeNet</td>
  <td>SSD</td>
  <td>SSD-GhostNet</td>
  <td>SSD-MobileNetV1-FPN</td>
  <td>SSD-MobileNetV2-FPNlite</td>
  <td>SSD-ResNet50</td>
</tr>
<tr>
  <td>SSD-ResNet50-FPN</td>
  <td>SSD-VGG16</td>
  <td>TextCNN</td>
  <td>TextRCNN</td>
  <td>TinyBert</td>
  <td>TinyDarknet</td>
</tr>
<tr>
  <td>Transformer</td>
  <td>UNet++</td>
  <td>UNet2D</td>
  <td>VGG16</td>
  <td>WarpCTC</td>
  <td>Wide&Deep</td>
</tr>
<tr>
  <td>WGAN</td>
  <td>Xception</td>
  <td>YOLOv3-DarkNet53</td>
  <td>YOLOv3-ResNet18</td>
  <td>YOLOv4</td>
  <td>YOLOv5</td>
</tr>
</table>

> In addition to the network in the above table, if the operator used in the user-defined network can be [exported to the MindIR model file](https://mindspore.cn/docs/programming_guide/en/master/save_model.html#export-mindir-model), the MindIR model file can also be used to execute inference tasks.
