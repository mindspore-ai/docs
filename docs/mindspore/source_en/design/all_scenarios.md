# Full-scenarios Unification

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/design/all_scenarios.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

MindSpore is designed to provide an device-edge-cloud full-scenarios AI framework that can be deployed in different hardware environments on the device-edge-cloud to meet the differentiated needs of different environments, such as supporting lightweight deployment on the device-side and rich training features such as automatic differentiation, hybrid precision and easy programming of models on the cloud side.

> The cloud side includes NVIDIA GPU, Huawei Ascend, Intel x86, etc., and the device side includes Arm, Qualcomm, Kirin, etc.

![intro](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindspore/source_zh_cn/design/images/all_scenarios_intro.png)

## Important Features of Full-scenarios

Several important features of MindSpore full scenarios:

1. The unified C++ inference interface of the device-edge-cloud supports algorithm code that can be quickly migrated to different hardware environments for execution, such as [device-side training based on C++ interface](https://mindspore.cn/lite/docs/en/master/quick_start/train_lenet.html).
2. Model unification. The device and cloud use the same model format and definition, and the software architecture is consistent. MindSpore supports the execution of Ascend, GPU, CPU (x86, Arm) and other hardware, and is used in multiple deployments for one training.
3. Diversified arithmetic support. Provide a unified southbound interface to support the quick addition of new hardware for use.
4. Model miniaturization techniques, adapted to the requirements of different hardware environments and business scenarios, such as quantization compression.
5. Rapid application of device-edge-cloud collaboration technologies such as [Federated Learning](https://mindspore.cn/federated/docs/en/master/index.html), [End-side Training](https://mindspore.cn/lite/docs/en/master/use/runtime_train.html) and other new technologies.

## Full-scenarios Support Mode

As shown above, the model files trained on MindSpore can be deployed in cloud services via Serving and executed on servers, device-side and other devices via Lite. Lite also supports offline optimization of the model via the standalone tool convert, achieving the goal of lightweighting the framework during inference and high performance of model execution.

MindSpore abstracts a uniform operator interface across hardware, so that the programming code for the network model can be consistent across different hardware environments. Simultaneously loading the same model file performs inference efficiently on each of the different hardware supported by MindSpore.

The inference aspect takes into account the fact that a large number of users use C++/C programming type, and provides a inference programming interface for C++, and the related programming interface is morphologically closer to the style of the Python interface.

At the same time, by providing custom offline optimized registration for third-party hardware and custom operator registration mechanism for third-party hardware, it implements fast interconnection to new hardware, while the external model programming interface and model files remain unchanged.
