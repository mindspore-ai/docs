# Inference Execution

Translator: [Dongdong92](https://gitee.com/zy179280)

<!-- TOC -->

- [Inference Execution](#inference-execution)
    - [Inference Service Based on Models](#inference-service-based-on-models)
        - [Overview](#overview)
        - [Executing Inference on Different Platforms](#executing-inference-on-different-platforms)
    - [On-line Inference Service Deployment Based on MindSpore Serving](#On-line-inference-service-deployment-based-on-mindspore-serving)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source_en.png"></a>

For trained models, MindSpore can execute inference tasks on different hardware platforms. MindSpore also provides online inference services based on MindSpore Serving.

## Inference Service Based on Models

### Overview

MindSpore supports to save files of [training parameters](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference.html#model-files) as CheckPoint format. MindSpore also supports to save [network model](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference.html#model-files) files as MindIR, AIR, and ONNX.

Referring to the [executing inference](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference.html#inference-execution) section, users not only can execute local inference through `mindspore.model.predict` interface, but also can export MindIR, AIR, and ONNX model files through `mindspore.export` for inference on different hardware platforms.

For dominating the difference between backend models, model files in the [MindIR format](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference.html#inference-execution) is recommended. MindIR model files can be executed on different hardware platforms, as well as be deployed to the Serving platform on cloud and the Lite platform on device.

### Executing Inference on Different Platforms

- For the Ascend hardware platform, please refer to [Inference on the Ascend 910 AI processor](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference_ascend_910.html) and [Inference on Ascend 310](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference_ascend_310.html).
- For the GPU hardware platform, please refer to [Inference on a GPU](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference_gpu.html).
- For the CPU hardware platform, please refer to [Inference on a CPU](https://www.mindspore.cn/docs/programming_guide/en/master/multi_platform_inference_cpu.html).
- For inference on the Lite platform on device, please refer to [on-device inference](https://www.mindspore.cn/lite/docs/en/master/index.html).

> Please refer to [MindSpore C++ Library Use](https://www.mindspore.cn/docs/faq/en/master/inference.html#c) to solve the interface issues on the Ascend hardware platform.

## On-line Inference Service Deployment Based on MindSpore Serving

MindSpore Serving is a lite and high-performance service module, aiming at assisting MindSpore developers in efficiently deploying on-line inference services. When a user completes the training task by using MindSpore, the trained model can be exported for inference service deployment through MindSpore Serving. Please refer to the following examples for deployment:

- [MindSpore Serving-based Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/master/serving_example.html).
- [gRPC-based MindSpore Serving Access](https://www.mindspore.cn/serving/docs/en/master/serving_grpc.html).
- [RESTful-based MindSpore Serving Access](https://www.mindspore.cn/serving/docs/en/master/serving_restful.html).
- [Servable Provided Through Model Configuration](https://www.mindspore.cn/serving/docs/en/master/serving_model.html).
- [MindSpore Serving-based Distributed Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/master/serving_distributed_example.html).

> For deployment issues regarding the on-line inference service, please refer to [MindSpore Serving](https://www.mindspore.cn/docs/faq/en/master/inference.html#mindspore-serving).
