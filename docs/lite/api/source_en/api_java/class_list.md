# Class List

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/api/source_en/api_java/class_list.md)

| Package                   | Class Name | Description                                              | Supported At Cloud-side Inference | Supported At Device-side Inference |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |--------|--------|
| com.mindspore        | [Model](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model.html) | Model defines model in MindSpore for compiling and running compute graph.  | √      | √      |
| com.mindspore.config | [MSContext](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mscontext.html) | MSContext defines for holding environment variables during runtime.                         | √      | √      |
| com.mindspore        | [MSTensor](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/mstensor.html) | MSTensor defines the tensor in MindSpore.          | √      | √      |
| com.mindspore        | [ModelParallelRunner](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/model_parallel_runner.html) | Defines MindSpore Lite concurrent inference.            | √      | ✕      |
| com.mindspore.config   | [RunnerConfig](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/runner_config.html) | RunnerConfig defines configuration parameters for concurrent inference.             | √      | ✕      |
| com.mindspore        | [Graph](https://www.mindspore.cn/lite/api/en/r2.6.0rc1/api_java/graph.html) | Graph defines the compute graph in MindSpore.           | ✕      | √      |
| com.mindspore.config | [CpuBindMode](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/CpuBindMode.java) | CpuBindMode defines the CPU binding mode.                                | √      | √      |
| com.mindspore.config | [DeviceType](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/DeviceType.java) | DeviceType defines the back-end device type.                                | √      | √      |
| com.mindspore.config  | [DataType](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/DataType.java) | DataType defines the supported data types.                             | √      | √      |
| com.mindspore.config   | [Version](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/Version.java) | Version is used to obtain the version information of MindSpore.                    | ✕      | √      |
| com.mindspore.config   | [ModelType](https://gitee.com/mindspore/mindspore/blob/v2.6.0-rc1/mindspore/lite/java/src/main/java/com/mindspore/config/ModelType.java) | ModelType defines the model file type.                  | √      | √      |
