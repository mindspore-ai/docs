# 推理执行

<!-- TOC -->

- [推理执行](#推理执行)
    - [基于模型推理服务](#基于模型推理服务)
        - [总览](#总览)
        - [不同硬件平台执行推理](#不同硬件平台执行推理)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.2/docs/migration_guide/source_zh_cn/inference.md" target="_blank"><img src="./_static/logo_source.png"></a>

MindSpore可以基于训练好的模型，在不同的硬件平台上执行推理任务。

## 基于模型推理服务

### 总览

MindSpore支持保存为CheckPoint格式的[训练参数文件](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/multi_platform_inference.html#id2)和MindIR、AIR、ONNX格式的[网络模型文件](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/multi_platform_inference.html#id2)。

参考[执行推理](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/multi_platform_inference.html#id3)，不仅可以直接通过`mindspore.model.predict`接口执行本机推理，还可以通过`mindspore.export`导出MindIR、AIR、ONNX格式的网络模型文件，以便于跨平台执行推理。

使用[MindIR格式](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/multi_platform_inference.html#id3)的模型文件消除了不同后端模型的差异，可以用于执行跨硬件平台推理，支持部署端侧Lite平台。

### 不同硬件平台执行推理

- Ascend硬件平台参考[Ascend 910 AI处理器上推理](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/multi_platform_inference_ascend_910.html)和[Ascend 310 AI处理器上推理](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/multi_platform_inference_ascend_310.html)。
- GPU硬件平台参考[GPU上推理](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/multi_platform_inference_gpu.html)。
- CPU硬件平台参考[CPU上推理](https://www.mindspore.cn/tutorial/inference/zh-CN/r1.2/multi_platform_inference_cpu.html)。
- Lite端侧推理的相关应用参考[端侧推理](https://www.mindspore.cn/lite/docs?master)。

> Ascend硬件平台推理的接口使用问题参考[C++接口使用类](https://www.mindspore.cn/doc/faq/zh-CN/r1.2/mindspore_cpp_library.html)解决。
