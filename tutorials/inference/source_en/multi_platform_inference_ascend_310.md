# Inference on the Ascend 310 AI processor

`Linux` `Ascend` `Inference Application` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/tutorials/inference/source_en/multi_platform_inference_ascend_310.md)

## Inference Using an ONNX or AIR File

The Ascend 310 AI processor is equipped with the ACL framework and supports the OM format which needs to be converted from the model in ONNX or AIR format. For inference on the Ascend 310 AI processor, perform the following steps:

1. Generate a model in ONNX or AIR format on the training platform. For details, see [Export AIR Model](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_model.html#export-air-model) and [Export ONNX Model](https://www.mindspore.cn/tutorial/training/en/r1.0/use/save_model.html#export-onnx-model).

2. Convert the ONNX or AIR model file into an OM model file and perform inference.
   - For performing inference in the cloud environment (ModelArt), see the [Ascend 910 training and Ascend 310 inference samples](https://support.huaweicloud.com/bestpractice-modelarts/modelarts_10_0026.html).
   - For details about the local bare-metal environment where the Ascend 310 AI processor is deployed in local (compared with the cloud environment), see the document of the Ascend 310 AI processor software package.
