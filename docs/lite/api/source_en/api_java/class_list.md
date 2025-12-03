# Class List

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://atomgit.com/mindspore/docs/blob/master/docs/lite/api/source_en/api_java/class_list.md)

| Package                   | Class Name | Description                                              | Supported At Cloud-side Inference | Supported At Device-side Inference |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |--------|--------|
| com.mindspore        | [Model](https://www.mindspore.cn/lite/api/en/master/api_java/model.html) | Model defines model in MindSpore for compiling and running compute graph.  | √      | √      |
| com.mindspore.config | [MSContext](https://www.mindspore.cn/lite/api/en/master/api_java/mscontext.html) | MSContext is used to save the context during execution.                         | √      | √      |
| com.mindspore        | [MSTensor](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html) | MSTensor defines the tensor in MindSpore.          | √      | √      |
| com.mindspore        | [ModelParallelRunner](https://www.mindspore.cn/lite/api/en/master/api_java/model_parallel_runner.html) | Defines MindSpore Lite concurrent inference.            | √      | ✕      |
| com.mindspore.config   | [RunnerConfig](https://www.mindspore.cn/lite/api/en/master/api_java/runner_config.html) | RunnerConfig defines configuration parameters for concurrent inference.             | √      | ✕      |
| com.mindspore        | [Graph](https://www.mindspore.cn/lite/api/en/master/api_java/graph.html) | Graph defines the compute graph in MindSpore.           | ✕      | √      |
| com.mindspore.config | [CpuBindMode](https://www.mindspore.cn/lite/api/en/master/api_java/mscontext.html#cpubindmode) | CpuBindMode defines the CPU binding mode.                                | √      | √      |
| com.mindspore.config | [DeviceType](https://www.mindspore.cn/lite/api/en/master/api_java/mscontext.html#devicetype) | DeviceType defines the back-end device type.                                | √      | √      |
| com.mindspore.config  | [DataType](https://www.mindspore.cn/lite/api/en/master/api_java/mstensor.html#datatype) | DataType defines the supported data types.                             | √      | √      |
| com.mindspore.config   | [Version](https://www.mindspore.cn/lite/api/en/master/api_java/version.html) | Version is used to obtain the version information of MindSpore.                    | ✕      | √      |
| com.mindspore.config   | [ModelType](https://www.mindspore.cn/lite/api/en/master/api_java/model.html#modeltype) | ModelType defines the model file type.                  | √      | √      |
| com.mindspore.config | [AscendDeviceInfo](https://www.mindspore.cn/lite/api/en/master/api_java/ascend_device_info.html) | The AscendDeviceInfo class is used to configure MindSpore Lite Ascend device options. | √ | ✕ |
| com.mindspore.config | [TrainCfg](https://www.mindspore.cn/lite/api/en/master/api_java/train_cfg.html) | Configuration parameters used for model training on the device. | ✕ | √ |

