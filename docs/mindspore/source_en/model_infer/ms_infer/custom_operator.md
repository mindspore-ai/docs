# Custom Operators

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.4.0/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.4.0/docs/mindspore/source_en/model_infer/ms_infer/custom_operator.md)

Large language model inference usually uses multiple optimization technologies, including but not limited to quantization and KVCache, to improve model efficiency and reduce the computing resources required. In addition to these common optimization technologies, you can customize the model structure based on specific application scenarios and requirements. Such customization may involve adding or deleting model layers, adjusting the connection mode, and even optimizing operators to achieve more efficient data processing and faster inference response. It enables the model to better adapt to specific tasks and running environments, but increases the complexity of the model.

Therefore, corresponding APIs are provided for you to develop custom operators and connect them to the MindSpore framework. Custom operators can achieve targeted performance optimizations. For example, through operator fusion technology, multiple operations can be combined into a single, more efficient operation, reducing I/O and delivery duration, while enhancing operator execution performance. This leads to in-depth optimization of the inference performance of the large language model.

For details about how to develop custom operators and effectively integrate them into the MindSpore framework, see [AOT-Type Custom Operators(Ascend)] (https://www.mindspore.cn/docs/en/r2.4.0/model_train/custom_program/operation/op_custom_ascendc.html).
