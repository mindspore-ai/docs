# 类列表

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_zh_cn/api_java/class_list.md)

| 包                        | 类                                                           | 描述                                                         | 云侧推理是否支持 | 端侧推理是否支持 |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |--------|--------|
| com.mindspore        | [Model](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model.html) | Model定义了MindSpore中的模型，用于计算图的编译和执行。 | √      | √      |
| com.mindspore.config | [MSContext](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mscontext.html) | MSContext用于保存执行期间的上下文。                         | √      | √      |
| com.mindspore        | [MSTensor](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/mstensor.html) | MSTensor定义了MindSpore中的张量。                       | √      | √      |
| com.mindspore        | [ModelParallelRunner](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/model_parallel_runner.html) | 定义了MindSpore Lite并发推理。                       | √      | ✕      |
| com.mindspore.config   | [RunnerConfig](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/runner_config.html) | RunnerConfig 定义并发推理的配置参数。                    | √      | ✕      |
| com.mindspore        | [Graph](https://www.mindspore.cn/lite/api/zh-CN/r2.6.0rc1/api_java/graph.html) | Model定义了MindSpore中的计算图。          | ✕      | √      |
| com.mindspore.config | [CpuBindMode](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/CpuBindMode.java) | CpuBindMode定义了CPU绑定模式。                               | √      | √      |
| com.mindspore.config | [DeviceType](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/DeviceType.java) | DeviceType定义了后端设备类型。                               | √      | √      |
| com.mindspore.config  | [DataType](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/DataType.java) | DataType定义了所支持的数据类型。                             | √      | √      |
| com.mindspore.config   | [Version](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/Version.java) | Version用于获取MindSpore的版本信息。                    | ✕      | √      |
| com.mindspore.config   | [ModelType](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/ModelType.java) | ModelType 定义了模型文件的类型。                    | √      | √      |
