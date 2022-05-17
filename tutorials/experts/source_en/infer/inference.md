# Inference Model Overview

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/infer/inference.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

MindSpore can execute inference tasks on different hardware platforms based on trained models.

Ascend 310 is an energy-efficient, highly integrated AI processor for edge scenarios that supports inference in Both MindIR and AIR format models.

MindIR format can be exported by MindSpore CPU, GPU, Ascend 910, can be run on GPU, Ascend 910, Ascend 310, no need to manually perform model conversion before inference. Inference needs to install MindSpore, and call MindSpore C++ API.

Only MindSpore Ascend 910 can export AIR format, and only Ascend 310 can be inferred. Before inference, you need to use the atc tool in Ascend CANN for model conversion. Inference does not rely on MindSpore, but only require the Ascend CANN package.

## Model Files

MindSpore can save two types of data: training parameters and network models that contain parameter information.

- Training parameters are stored in the Checkpoint format.
- Network models are stored in the MindIR, AIR, or ONNX format.

Basic concepts and application scenarios of these formats are as follows:

- Checkpoint
    - Checkpoint uses the Protocol Buffers format and stores all network parameter values.
    - It is generally used to resume training after a training task is interrupted or execute a fine-tune (Fine Tune) task after training.
- MindIR
    - MindSpore IR is a functional IR based on graph representation of MindSpore, and defines the extensible graph structure and ir representation of the operator.
    - It eliminates model differences between different backends and is generally used to perform inference tasks across hardware platforms.
- ONNX
    - Open Neural Network Exchange is an open format built to represent machine learning models.
    - It is generally used to transfer models between different frameworks or used on the inference engine ([TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/python_api/index.html)).
    - At present, MindSpore only supports the export of ONNX model, and does not support loading ONNX model for inference. Currently, the models supported for export are Resnet50, YOLOv3_darknet53, YOLOv4 and BERT. These models can be used on [ONNX Runtime](https://onnxruntime.ai/).
- AIR
    - Ascend Intermediate Representation is an open file format defined by Huawei for machine learning.
    - It adapts to Huawei AI processors well and is generally used to execute inference tasks on Ascend 310.

## Inference Execution

Inference can be classified into the following two modes based on the application environment:

1. Local inference

    Load a checkpoint file generated during network training and call the `model.predict` API for inference and validation.

2. Cross-platform inference

    Use a network definition and a checkpoint file, call the `export` API to export a model file, and perform inference on different platforms. Currently, MindIR, ONNX, and AIR (on only Ascend AI Processors) models can be exported. For details, see [Saving Models](https://www.mindspore.cn/tutorials/en/master/advanced/train/save.html).

## Introduction to MindIR

MindSpore defines logical network structures and operator attributes through a unified IR, and decouples model files in MindIR format from hardware platforms to implement one-time training and multiple-time deployment.

1. Overview

    As a unified model file of MindSpore, MindIR stores network structures and weight parameter values. In addition, it can be deployed on the on-cloud Serving and the on-device Lite platforms to execute inference tasks.

    A MindIR file supports the deployment of multiple hardware forms.

    - On-cloud deployment and inference on Serving: After MindSpore trains and generates a MindIR model file, the file can be directly sent to MindSpore Serving for loading and inference. No additional model conversion is required. This ensures that models on different hardware such as Ascend, GPU, and CPU are unified.
    - On-device inference and deployment on Lite: MindIR can be directly used for Lite deployment. In addition, to meet the lightweight requirements on devices, the model miniaturization and conversion functions are provided. An original MindIR model file can be converted from the Protocol Buffers format to the FlatBuffers format for storage, and the network structure is lightweight to better meet the performance and memory requirements on devices.

2. Application Scenarios

    Use a network definition and a checkpoint file to export a MindIR model file, and then execute inference based on different requirements, for example, [Inference Using the MindIR Model on Ascend 310 AI Processors](https://www.mindspore.cn/tutorials/experts/en/master/infer/ascend_310_mindir.html), [MindSpore Serving-based Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/master/serving_example.html), and [Inference on Devices](https://www.mindspore.cn/lite/docs/en/master/index.html).

## model.eval Model Validation

### Model Saved Locally

First build the model, then use the `mindspore` module's `load_checkpoint` and `load_param_into_net` to load the model and parameters from the local, and pass in the validation dataset to perform model inference. The validation dataset is processed in the same way as the training dataset.

```python
network = LeNet5(cfg.num_classes)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
model = Model(network, net_loss, metrics={"Accuracy": Accuracy()})

print("============== Starting Testing ==============")
param_dict = load_checkpoint(args.ckpt_path)
load_param_into_net(network, param_dict)
dataset = create_dataset(os.path.join(args.data_path, "test"),
                            cfg.batch_size,)
acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
print("============== {} ==============".format(acc))
```

Where

`model.eval` is the model validation interface, and corresponding interface description is as follows: <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.Model.eval>.

> Inference sample code: <https://gitee.com/mindspore/models/blob/master/official/cv/lenet/eval.py>.

### Using MindSpore Hub to Load Models from HUAWEI CLOUD

First build the model, then use `mindspore_hub.load` to load the model parameters from the cloud, and pass in the validation dataset to infer. The validation dataset is processed in the same way as the training dataset.

```python
model_uid = "mindspore/ascend/0.7/googlenet_v1_cifar10"  # using GoogleNet as an example.
network = mindspore_hub.load(model_uid, num_classes=10)
net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
model = Model(network, net_loss, metrics={"Accuracy": Accuracy()})

print("============== Starting Testing ==============")
dataset = create_dataset(os.path.join(args.data_path, "test"),
                            cfg.batch_size,)
acc = model.eval(dataset, dataset_sink_mode=args.dataset_sink_mode)
print("============== {} ==============".format(acc))
```

Where

`mindspore_hub.load` is the interface for loading model parameter, and corresponding interface description is as follows: <https://www.mindspore.cn/hub/docs/zh-CN/master/hub.html#mindspore-hubload>.

## Using the `model.predict` Interface for Inference Operations

```python
model.predict(input_data)
```

Where

`model.predict` is inference interface, and corresponding interface description is as follows: <https://www.mindspore.cn/docs/en/master/api_python/mindspore.html#mindspore.Model.predict>.