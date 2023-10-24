# Inference Model Overview

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.2/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.2/tutorials/experts/source_en/infer/inference.md)

MindSpore can execute inference tasks on different hardware platforms based on trained models.

Ascend 310 is an energy-efficient, highly integrated AI processor for edge scenarios that supports inference in Both MindIR format models.

MindIR format can be exported by MindSpore CPU, GPU, Ascend 910, can be run on GPU, Ascend 910, Ascend 310, no need to manually perform model conversion before inference. Inference needs to install MindSpore Lite, and call MindSpore Lite C++ API.

## Model Files

MindSpore can save two types of data: training parameters and network models that contain parameter information.

- Training parameters are stored in the Checkpoint format.
- Network models are stored in the MindIR, or ONNX format.

Basic concepts and application scenarios of these formats are as follows:

- Checkpoint
    - Checkpoint uses the Protocol Buffers format and stores all network parameter values.
    - It is generally used to resume training after a training task is interrupted or execute a fine-tune (Fine Tune) task after training.
- MindIR
    - MindSpore IR is a functional IR based on graph representation of MindSpore, and defines the extensible graph structure and ir representation of the operator.
    - It eliminates model differences between different backends and is generally used to perform inference tasks across hardware platforms.
- ONNX
    - Open Neural Network Exchange is an open format built to represent machine learning models.
    - It is generally used to transfer models between different frameworks or used on the inference engine ([TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)).
    - Currently, the models supported for export are Resnet50, YOLOv3_darknet53, YOLOv4 and BERT. These models can be used on [ONNX Runtime](https://onnxruntime.ai/).
    - Inference execution is achieved on Ascend via [ACT tool](https://www.hiascend.com/document/detail/zh/canncommercial/63RC1/inferapplicationdev/aclcppdevg/aclcppdevg_000001.html).

## Inference Execution

Inference can be classified into the following two modes based on the application environment:

1. Local inference

    Load a checkpoint file generated during network training and call the `model.predict` API for inference and validation.

2. Cross-platform inference

    Use a network definition and a checkpoint file, call the `export` API to export a model file, and perform inference on different platforms. Currently, MindIR and ONNX (on only Ascend AI Processors) models can be exported. For details, see [Saving Models](https://www.mindspore.cn/tutorials/en/r2.2/beginner/save_load.html).

## Introduction to MindIR

MindSpore defines logical network structures and operator attributes through a unified IR, and decouples model files in MindIR format from hardware platforms to implement one-time training and multiple-time deployment.

1. Overview

    As a unified model file of MindSpore, MindIR stores network structures and weight parameter values. In addition, it can be deployed on the on-cloud Serving and the MindSpore Lite platforms to execute inference tasks.

    A MindIR file supports the deployment of multiple hardware forms.

    - On-cloud deployment and inference on Serving: After MindSpore trains and generates a MindIR model file, the file can be directly sent to MindSpore Serving for loading and inference. No additional model conversion is required. This ensures that models on different hardware such as Ascend, GPU, and CPU are unified.
    - Use MindSpore Lite for inference and deployment: MindIR models can be deployed by directly using Lite. Support deployment on on-cloud servers such as Ascend, Nvidia GPUs, CPUs, as well as on resource-constrained on-device hardware such as cell phones.

2. Application Scenarios

    Use a network definition and a checkpoint file to export a MindIR model file, and then execute inference based on different requirements, for example, [MindSpore Serving-based Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/r2.0/serving_example.html) and [Lite Inference](https://www.mindspore.cn/lite/docs/en/r2.2/index.html).

## model.eval Model Validation

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

`model.eval` is the model validation interface, and corresponding interface description is as follows: [mindspore.train.Model](https://www.mindspore.cn/docs/en/r2.2/api_python/train/mindspore.train.Model.html#mindspore.train.Model).

> Inference sample code: [eval.py](https://gitee.com/mindspore/models/blob/r2.2/research/cv/lenet/eval.py).

## Using the `model.predict` Interface for Inference Operations

```python
model.predict(input_data)
```

Where

`model.predict` is inference interface, and corresponding interface description is as follows: [mindspore.train.Model.predict](https://www.mindspore.cn/docs/en/r2.2/api_python/train/mindspore.train.Model.html#mindspore.train.Model.predict).
