# Image Classification Model

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/lite/docs/source_en/reference/image_classification_lite.md)

## Image Classification Introduction

Image classification model is to identity what an image represents, to predict the object list and the probabilities. For example, the following table shows the classification results after mode inference.

![image_classification](images/image_classification_result.png)

| Category   | Probability |
| ---------- | ----------- |
| Flower      | 0.9620 |
| Plant       | 0.9216 |
| Petal       | 0.7893 |
| Flowerpot   | 0.6171 |

Use MindSpore Lite to realize image classification [example](https://gitee.com/mindspore/models/tree/master/official/lite/image_classification).

## Image Classification Model List

The following table shows the data of some image classification models using MindSpore Lite inference.

> The performance of the table below is tested on the mate30.

| Model name         | Size(Mb) | Top1 | Top5 | F1 | CPU 4 thread delay (ms) |
|-----------------------| :----------: | :----------: | :----------: | :----------: | :-----------: |
| [MobileNetV2](https://download.mindspore.cn/model_zoo/official/lite/mobilenetv2_openimage_lite/mobilenetv2.ms) | 11.5 | - | - | 65.5% | 14.595 |
| [Inceptionv3](https://download.mindspore.cn/model_zoo/official/lite/inceptionv3_lite/inceptionv3.ms) | 90.9 | 78.62% | 94.08% | - | 92.086 |
| [Shufflenetv2](https://download.mindspore.cn/model_zoo/official/lite/shufflenetv2_lite/shufflenetv2.ms) | 8.8 | 67.74% | 87.62% | - | 8.303 |
| [GoogleNet](https://download.mindspore.cn/model_zoo/official/lite/googlenet_lite/googlenet.ms) | 25.3 | 72.2% | 90.06% | - | 23.257 |
| [ResNext50](https://download.mindspore.cn/model_zoo/official/lite/resnext50_lite/resnext50.ms) | 95.8 | 73.1% | 91.21% | - | 138.164 |
| [GhostNet](https://download.mindspore.cn/model_zoo/official/lite/ghostnet_lite/ghostnet.ms) | 15.0 | 73.9% | 91.40% | - | 9.959 |
| [GhostNet600](https://download.mindspore.cn/model_zoo/official/lite/ghostnet_lite/ghostnet600.ms) | 40.4 | 80.2% | 94.90% | - | 52.243 |
| [GhostNet_int8](https://download.mindspore.cn/model_zoo/official/lite/ghostnet_lite/ghostnet_int8.ms) | 15.3 | 73.6% | - | - | 31.452 |
| [VGG-Small-low_bit](https://download.mindspore.cn/model_zoo/official/lite/low_bit_quant/low_bit_quant_bs_1.ms) | 17.8 | 93.7% | - | - | 9.082 |
| [ResNet50-0.65x](https://download.mindspore.cn/model_zoo/official/lite/adversarial_pruning_lite/adversarial_pruning.ms) | 48.6 | 80.2% | - | - | 89.816 |
| [plain-CNN-ResNet18](https://download.mindspore.cn/model_zoo/official/lite/residual_distill_lite/residual_distill_res18_cifar10_bs_1_update.ms) | 97.3 | 95.4% | - | - | 63.227 |
| [plain-CNN-ResNet34](https://download.mindspore.cn/model_zoo/official/lite/residual_distill_lite/residual_distill_res34_cifar10_bs_1_update.ms) | 80.5 | 95.0% | - | - | 20.652 |
| [plain-CNN-ResNet50](https://download.mindspore.cn/model_zoo/official/lite/residual_distill_lite/residual_distill_res50_cifar10_bs_1_update.ms) | 89.6 | 94.5% | - | - | 24.561 |
