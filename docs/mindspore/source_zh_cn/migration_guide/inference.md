# 推理执行

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.8/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.8/docs/mindspore/source_zh_cn/migration_guide/inference.md)

MindSpore可以基于训练好的模型，在不同的硬件平台上执行推理任务，还可以基于MindSpore Serving部署在线推理服务。

## 基于模型推理服务

### 总览

MindSpore支持保存为CheckPoint格式的[训练参数文件](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/infer/inference.html#模型文件)和MindIR、AIR、ONNX格式的[网络模型文件](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/infer/inference.html#模型文件)。

参考[执行推理](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/infer/inference.html#执行推理)，不仅可以直接通过`mindspore.model.predict`接口执行本机推理，还可以通过`mindspore.export`导出MindIR、AIR、ONNX格式的网络模型文件，以便于跨平台执行推理。

使用[MindIR格式](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/infer/inference.html#mindir介绍)的模型文件消除了不同后端模型的差异，可以用于执行跨硬件平台推理，支持部署到云端Serving和端侧Lite平台。

### 不同硬件平台执行推理

- Ascend硬件平台参考[Ascend 910 AI处理器上推理](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/infer/ascend_910_mindir.html)和[Ascend 310 AI处理器上推理](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/infer/ascend_310_air.html)。
- GPU/CPU硬件平台参考[GPU/CPU上推理](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/infer/cpu_gpu_mindir.html)。
- Lite端侧推理的相关应用参考[端侧推理](https://www.mindspore.cn/lite/docs/zh-CN/r1.8/index.html)。

> Ascend硬件平台推理的接口使用问题参考[C++接口使用类](https://www.mindspore.cn/docs/zh-CN/r1.8/faq/inference.html)解决。

## 基于MindSpore Serving部署在线推理服务

MindSpore Serving是一个轻量级、高性能的服务模块，旨在帮助MindSpore开发者在生产环境中高效部署在线推理服务。当用户使用MindSpore完成模型训练后，导出MindSpore模型，即可使用MindSpore Serving创建该模型的推理服务。参考以下几个样例进行部署：

- [基于MindSpore Serving部署推理服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.8/serving_example.html)。
- [基于gRPC接口访问MindSpore Serving服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.8/serving_grpc.html)。
- [基于RESTful接口访问MindSpore Serving服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.8/serving_restful.html)。
- [通过配置模型提供Servable](https://www.mindspore.cn/serving/docs/zh-CN/r1.8/serving_model.html)。
- [基于MindSpore Serving部署分布式推理服务](https://www.mindspore.cn/serving/docs/zh-CN/r1.8/serving_distributed_example.html)。

> MindSpore Serving部署在线推理服务的问题可以参考[MindSpore Serving类](https://www.mindspore.cn/serving/docs/en/r1.8/faq.html)解决。
