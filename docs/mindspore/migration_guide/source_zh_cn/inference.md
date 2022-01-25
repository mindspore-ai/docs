# 推理执行

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_zh_cn/inference.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

MindSpore可以基于训练好的模型，在不同的硬件平台上执行推理任务，还可以基于MindSpore Serving部署在线推理服务。

## 基于模型推理服务

### 总览

MindSpore支持保存为CheckPoint格式的[训练参数文件](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference.html#模型文件)和MindIR、AIR、ONNX格式的[网络模型文件](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference.html#模型文件)。

参考[执行推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference.html#执行推理)，不仅可以直接通过`mindspore.model.predict`接口执行本机推理，还可以通过`mindspore.export`导出MindIR、AIR、ONNX格式的网络模型文件，以便于跨平台执行推理。

使用[MindIR格式](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference.html#mindir介绍)的模型文件消除了不同后端模型的差异，可以用于执行跨硬件平台推理，支持部署到云端Serving和端侧Lite平台。

### 不同硬件平台执行推理

- Ascend硬件平台参考[Ascend 910 AI处理器上推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_ascend_910.html)和[Ascend 310 AI处理器上推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_ascend_310.html)。
- GPU硬件平台参考[GPU上推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_gpu.html)。
- CPU硬件平台参考[CPU上推理](https://www.mindspore.cn/docs/programming_guide/zh-CN/master/multi_platform_inference_cpu.html)。
- Lite端侧推理的相关应用参考[端侧推理](https://www.mindspore.cn/lite/docs/zh-CN/master/index.html)。

> Ascend硬件平台推理的接口使用问题参考[C++接口使用类](https://www.mindspore.cn/docs/faq/zh-CN/master/inference.html#c)解决。

## 基于MindSpore Serving部署在线推理服务

MindSpore Serving是一个轻量级、高性能的服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。参考以下几个样例进行部署：

- [基于MindSpore Serving部署推理服务](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_example.html)。
- [基于gRPC接口访问MindSpore Serving服务](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_grpc.html)。
- [基于RESTful接口访问MindSpore Serving服务](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_restful.html)。
- [通过配置模型提供Servable](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_model.html)。
- [基于MindSpore Serving部署分布式推理服务](https://www.mindspore.cn/serving/docs/zh-CN/master/serving_distributed_example.html)。

> MindSpore Serving部署在线推理服务的问题可以参考[MindSpore Serving类](https://www.mindspore.cn/docs/faq/zh-CN/master/inference.html#mindspore-serving)解决。
