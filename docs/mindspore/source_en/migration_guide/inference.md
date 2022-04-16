# Inference Execution

Translator: [Dongdong92](https://gitee.com/zy179280)

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/migration_guide/inference.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

For trained models, MindSpore can execute inference tasks on different hardware platforms. MindSpore also provides online inference services based on MindSpore Serving.

## Inference Service Based on Models

### Overview

MindSpore supports to save [training parameters files](https://www.mindspore.cn/tutorials/experts/en/r1.7/infer/inference.html#model-files) as CheckPoint format. MindSpore also supports to save [network model files](https://www.mindspore.cn/tutorials/experts/en/r1.7/infer/inference.html#model-files) as MindIR, AIR, and ONNX.

Referring to the [executing inference](https://www.mindspore.cn/tutorials/experts/en/r1.7/infer/inference.html#inference-execution) section, users not only can execute local inference through `mindspore.model.predict` interface, but also can export MindIR, AIR, and ONNX model files through `mindspore.export` for inference on different hardware platforms.

For dominating the difference between backend models, model files in the [MindIR format](https://www.mindspore.cn/tutorials/experts/en/r1.7/infer/inference.html#inference-execution) is recommended. MindIR model files can be executed on different hardware platforms, as well as be deployed to the Serving platform on cloud and the Lite platform on device.

### Executing Inference on Different Platforms

- For the Ascend hardware platform, please refer to [Inference on the Ascend 910 AI processor](https://www.mindspore.cn/tutorials/experts/en/r1.7/infer/ascend_910_mindir.html) and [Inference on Ascend 310](https://www.mindspore.cn/tutorials/experts/en/r1.7/infer/ascend_310_mindir.html).
- For the GPU/CPU hardware platform, please refer to [Inference on a GPU/CPU](https://www.mindspore.cn/tutorials/experts/en/r1.7/infer/cpu_gpu_mindir.html).
- For inference on the Lite platform on device, please refer to [on-device inference](https://www.mindspore.cn/lite/docs/en/r1.7/index.html).

> Please refer to [MindSpore C++ Library Use](https://www.mindspore.cn/docs/en/r1.7/faq/inference.html) to solve the interface issues on the Ascend hardware platform.

## On-line Inference Service Deployment Based on MindSpore Serving

MindSpore Serving is a lite and high-performance service module, aiming at assisting MindSpore developers in efficiently deploying on-line inference services in the production environment. When a user completes the training task by using MindSpore, the trained model can be exported for inference service deployment via MindSpore Serving. Please refer to the following examples for deployment:

- [MindSpore Serving-based Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/r1.7/serving_example.html).
- [gRPC-based MindSpore Serving Access](https://www.mindspore.cn/serving/docs/en/r1.7/serving_grpc.html).
- [RESTful-based MindSpore Serving Access](https://www.mindspore.cn/serving/docs/en/r1.7/serving_restful.html).
- [Servable Provided Through Model Configuration](https://www.mindspore.cn/serving/docs/en/r1.7/serving_model.html).
- [MindSpore Serving-based Distributed Inference Service Deployment](https://www.mindspore.cn/serving/docs/en/r1.7/serving_distributed_example.html).

> For deployment issues regarding the on-line inference service, please refer to [MindSpore Serving Class](https://www.mindspore.cn/serving/docs/en/r1.7/faq.html).
