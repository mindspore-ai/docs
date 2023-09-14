# 官方模型库

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_zh_cn/note/official_models.md)

## 领域套件与扩展包

### 计算机视觉

#### 图像分类（骨干类）

以下数据基于Ascend 910环境和ImageNet-1K数据集获得。

| model | acc@1 | mindcv recipe | vanilla mindspore |
| :-:     | :-:        | :-:    | :-:  |
|  vgg11           |  71.86   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   |
|  vgg13           |  72.87   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   |
|  vgg16           |  74.61   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/VGG/vgg16) |
|  vgg19           |  75.21   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/VGG/vgg19) |
| resnet18         |  70.21   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/ResNet)  |
| resnet34         |  74.15   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/ResNet)   |
| resnet50         |  76.69   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/ResNet)    |
| resnet101        |  78.24   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/ResNet)    |
| resnet152        |  78.72   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/ResNet)    |
| resnetv2_50      |  76.90   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnetv2)      |   |
| resnetv2_101     |  78.48   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnetv2)      |   |
| dpn92            |  79.46   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |   |
| dpn98            |  79.94   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |    |
| dpn107           |  80.05   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |   |
| dpn131           |  80.07   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |   |
| densenet121      |  75.64   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |   |
| densenet161      |  79.09   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |   |
| densenet169      |  77.26   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |   |
| densenet201      |  78.14   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |   |
| seresnet18       |  71.81   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |   |
| seresnet34       |  75.36   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |   |
| seresnet50       |  78.31   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |   |
| seresnext26      |  77.18   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |   |
| seresnext50      |  78.71   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |   |
| skresnet18       |  73.09   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet)              |   |
| skresnet34       |  76.71   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet)              |   |
| skresnet50_32x4d |  79.08   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet)              |   |
| resnext50_32x4d  |  78.53   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |   |
| resnext101_32x4d |  79.83   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |   |
| resnext101_64x4d |  80.30   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |   |
| resnext152_64x4d |  80.52   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |   |
| rexnet_x09       |  77.07   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |   |
| rexnet_x10       |  77.38   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |   |
| rexnet_x13       |  79.06   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |   |
| rexnet_x15       |  79.94   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |   |
| rexnet_x20       |  80.64   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |   |
| resnest50        |  80.81   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnest)            |   |
| resnest101       |  82.50   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnest)            |   |
| res2net50        |  79.35   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |   |
| res2net101       |  79.56   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |   |
| res2net50_v1b    |  80.32   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |   |
| res2net101_v1b   |  95.41   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |   |
| googlenet        |  72.68   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/googlenet)          |   |
| inceptionv3      |  79.11   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/inceptionv3)        | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/Inception/inceptionv3) |
| inceptionv4      |  80.88   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/inceptionv4)        | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/Inception/inceptionv4) |
| mobilenet_v1_025 |  53.87   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v1_050 |  65.94   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v1_075 |  70.44   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v1_100 |  72.95   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v2_075 |  69.98   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |   |
| mobilenet_v2_100 |  72.27   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |   |
| mobilenet_v2_140 |  75.56   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |   |
| mobilenet_v3_small     | 68.10 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3)      |    |
| mobilenet_v3_large     | 75.23 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3)      | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/MobileNet/mobilenetv3) |
| shufflenet_v1_g3_x0_5  | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv1)     |    |
| shufflenet_v1_g3_x1_5  | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv1)     | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/ShuffleNet/shufflenetv1) |
| shufflenet_v2_x0_5     | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |    |
| shufflenet_v2_x1_0     | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/ShuffleNet/shufflenetv2) |
| shufflenet_v2_x1_5     | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |    |
| shufflenet_v2_x2_0     | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |    |
| xception               | 79.01 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/xception)         | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/Inception/xception) |
| ghostnet_50            | 66.03 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |   |
| ghostnet_100           | 73.78 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |   |
| ghostnet_130           | 75.50 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |   |
| nasnet_a_4x1056        | 73.65 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/nasnet)           |   |
| mnasnet_0.5            | 68.07 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| mnasnet_0.75           | 71.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| mnasnet_1.0            | 74.28 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| mnasnet_1.4            | 76.01 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| efficientnet_b0        | 76.89 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/efficientnet)     | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/Efficientnet/efficientnet-b0)
| efficientnet_b1        | 78.95 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/efficientnet)     | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/Efficientnet/efficientnet-b1)
| efficientnet_b2        | 79.80 |     | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/Efficientnet/efficientnet-b2) |
| efficientnet_b3        | 80.50 |     | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/Efficientnet/efficientnet-b3) |
| efficientnet_v2        | 83.77 |     | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/Efficientnet/efficientnetv2) |
| regnet_x_200mf         | 68.74 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_x_400mf         | 73.16 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_x_600mf         | 73.34 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_x_800mf         | 76.04 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_y_200mf         | 70.30 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_y_400mf         | 73.91 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_y_600mf         | 75.69 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| regnet_y_800mf         | 76.52 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |   |
| mixnet_s               | 75.52 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet)     |   |
| mixnet_m               | 76.64 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet)     |   |
| mixnet_l               | 78.73 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet)     |   |
| hrnet_w32              | 80.64 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/hrnet)      |   |
| hrnet_w48              | 81.19 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/hrnet)      |   |
| bit_resnet50           | 76.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit)         |   |
| bit_resnet50x3         | 80.63 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit)        |   |
| bit_resnet101          | 77.93 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit)        |   |
| repvgg_a0              | 72.19 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_a1              | 74.19 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_a2              | 76.63 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_b0              | 74.99 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_b1              | 78.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_b2              | 79.29 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |   |
| repvgg_b3            | 80.46 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)       |   |
| repvgg_b1g2          | 78.03 |[config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)        |   |
| repvgg_b1g4          | 77.64 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)       |   |
| repvgg_b2g4          | 78.80 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)       |   |
| repmlp_t224          | 76.71 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repmlp)       |   |
| convnext_tiny        | 81.91 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext)     |   |
| convnext_small       | 83.40 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext)     |   |
| convnext_base        | 83.32 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext)     |   |
| vit_b_32_224         | 75.86 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)          |  [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/VIT) |
| vit_l_16_224         | 76.34| [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)           |   |
| vit_l_32_224         | 73.71 |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)         |   |
| swintransformer_tiny | 80.82 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/swintransformer) | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/SwinTransformer) |
| pvt_tiny            | 74.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |   |
| pvt_small           | 79.66 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |   |
| pvt_medium          | 81.82 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |   |
| pvt_large           | 81.75 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |   |
| pvt_v2_b0           | 71.50 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |   |
| pvt_v2_b1           | 78.91 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |   |
| pvt_v2_b2           | 81.99 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |   |
| pvt_v2_b3           | 82.84 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |   |
| pvt_v2_b4           | 83.14 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |   |
| pit_ti              | 72.96 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |   |
| pit_xs              | 78.41 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |   |
| pit_s               | 80.56 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |   |
| pit_b               | 81.87 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |   |
| coat_lite_tiny      | 77.35 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat)          |   |
| coat_lite_mini      | 78.51 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat)          |   |
| coat_tiny           | 79.67 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat)          |   |
| convit_tiny         | 73.66 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| convit_tiny_plus    | 77.00 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| convit_small        | 81.63 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| convit_small_plus   | 81.80 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| convit_base         | 82.10 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| convit_base_plus    | 81.96 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |   |
| crossvit_9          | 73.56 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit)      |   |
| crossvit_15         | 81.08 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit)      |   |
| crossvit_18         | 81.93 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit)      |   |
| mobilevit_xx_small  | 68.90 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit)     |   |
| mobilevit_x_small   | 74.98 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit)     |   |
| mobilevit_small     | 78.48 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit)     |   |
| visformer_tiny      | 78.28 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |   |
| visformer_tiny_v2   | 78.82 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |   |
| visformer_small     | 81.76 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |   |
| visformer_small_v2  | 82.17 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |   |
| edgenext_xx_small   | 71.02 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |   |
| edgenext_x_small    | 75.14 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |   |
| edgenext_small      | 79.15 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |   |
| edgenext_base       | 82.24 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |   |
| poolformer_s12      | 77.33 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/poolformer)    |   |
| xcit_tiny_12_p16    | 77.67 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/xcit)          |   |

### 目标检测

以下数据基于Ascend 910环境和COCO2017数据集获得。

#### yolo

| model | map |  mindyolo recipe | vanilla mindspore |
|:-: | :-: | :-: | :-: |
| yolov8_n | 37.2 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |    |
| yolov8_s | 44.6 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |   |
| yolov8_m | 50.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |   |
| yolov8_l | 52.8 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |   |
| yolov8_x | 53.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |   |
| yolov7_t | 37.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |    |
| yolov7_l | 50.8 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |   |
| yolov7_x |  52.4| [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |   |
| yolov5_n | 27.3 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |   |
| yolov5_s | 37.6 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/YOLOv5) |
| yolov5_m | 44.9 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |   |
| yolov5_l | 48.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |   |
| yolov5_x | 50.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |   |
| yolov4_csp       | 45.4 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) |   |
| yolov4_csp(silu) | 45.8 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/YOLOv4) |
| yolov3_darknet53 | 45.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov3) | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/YOLOv3) |
| yolox_n | 24.1 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_t | 33.3 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_s | 40.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_m | 46.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_l | 49.2 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_x | 51.6 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_darknet53 | 47.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |

#### 经典

| model |  map | mind_series recipe | vanilla mindspore |
| :-:     |  :-:            | :-:        |  :-:  |
|  ssd_vgg16                 | 23.2  |   | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/SSD)|
|  ssd_mobilenetv1           | 22.0  |   | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/SSD)|
|  ssd_mobilenetv2           | 29.1  |   | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/SSD)|
|  ssd_resnet50              | 34.3  |   | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/SSD)|
|  fasterrcnn                  | 58    | [link](https://github.com/mindspore-lab/models/tree/master/official/cv/RCNN)  | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/FasterRCNN) |
|  maskrcnn_mobilenetv1      | coming soon   |   | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/MaskRCNN/maskrcnn_mobilenetv1) |
|  maskrcnn_resnet50         | coming soon   |   | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/MaskRCNN/maskrcnn_resnet50) |

### 语义分割

| model |  mind_series recipe | vanilla mindspore |
| :-:     |  :-:            | :-:     |
| ocrnet          |   [link](https://github.com/mindspore-lab/models/tree/master/official/cv/OCRNet)   | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/OCRNet/)         |
| deeplab v3      |      | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/DeepLabv3)       |
| deeplab v3 plus |      | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/DeepLabV3P)      |
| unet            |      | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/Unet)            |

### OCR

#### 文本检测

| model  |dataset | fscore | mindocr recipe | vanilla mindspore |
:-:     |   :-:       | :-:        | :-:   |  :-:   |
| dbnet_mobilenetv3  | icdar2015          | 77.23 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/DBNet/)  |
| dbnet_resnet18     | icdar2015          | 81.73 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/DBNet/)  |
| dbnet_resnet50     | icdar2015          | 85.05 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/DBNet/)  |
| dbnet++_resnet50   | icdar2015          | 86.74 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |   |
| psenet_resnet152   | icdar2015          | 82.06 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) | [link](https://gitee.com/mindspore/models/tree/r2.1/research/cv/psenet)  |
| east_resnet50      | icdar2015          | 84.87 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)   | [link](https://gitee.com/mindspore/models/tree/r2.1/research/cv/east)    |
| fcenet_resnet50    | icdar2015          | 84.12 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/fcenet)   |   |

#### 文本识别

| model | dataset | acc | mindocr recipe | vanilla mindspore |
:-:     |   :-:       | :-:        | :-:   |  :-:   |
| svtr_tiny          | IC03,13,15,IIIT,etc | 89.02 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/svtr)   |   |
| crnn_vgg7          | IC03,13,15,IIIT,etc | 82.03 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/CRNN)    |
| crnn_resnet34_vd   | IC03,13,15,IIIT,etc | 84.45 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |   |
| rare_resnet34_vd   | IC03,13,15,IIIT,etc | 85.19 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare)   | [link](https://gitee.com/mindspore/models/tree/r2.1/research/cv/crnn_seq2seq_ocr)  |

#### 文本方向分类

| model | dataset | acc | mindocr recipe |
:-:     |   :-:       | :-:        | :-:   |
| mobilenetv3  | RCTW17,MTWI,LSVT | 94.59 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/cls/mobilenetv3)   |

### 人脸

| model | dataset | acc | mindface recipe | vanilla mindspore
| :-:     |  :-:       | :-:        | :-:   | :-: |
| arcface_mobilefacenet-0.45g  | MS1MV2          | 98.70  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |   |
| arcface_r50                  | MS1MV2          | 99.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| arcface_r100                 | MS1MV2          | 99.38  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/Arcface) |
| arcface_vit_t                | MS1MV2          | 99.71  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |   |
| arcface_vit_s                | MS1MV2          | 99.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| arcface_vit_b                | MS1MV2          | 99.81  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| arcface_vit_l                | MS1MV2          | 99.75  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| retinaface_mobilenet_0.25    | WiderFace        | 90.77/88.2/74.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/detection)  | [link](https://gitee.com/mindspore/models/tree/r2.1/research/cv/retinaface) |
| retinaface_r50               | WiderFace        | 95.07/93.61/84.84 | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/detection)  | [link](https://gitee.com/mindspore/models/tree/r2.1/official/cv/RetinaFace_ResNet50) |

### 语言模型

| model |  mindformer recipe | vanilla mindspore
| :-:     |  :-:   | :-: |
| bert_base   | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/t5.md) | [link](https://gitee.com/mindspore/models/tree/r2.1/official/nlp/Bert) |
| t5_small    | [config](https://github.com/mindspore-lab/mindformers/blob/master/docs/model_cards/bert.md) |  |
| gpt2_small  | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md) |
| gpt2_13b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md) |
| gpt2_52b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/gpt2.md) |
| pangu_alpha | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/pangualpha.md) |
| glm_6b       | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm.md)  |
| glm_6b_lora  | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/glm.md)  |
| llama_7b     | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md) |
| llama_13b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md) |
| llama_65b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md) |
| llama_7b_lora | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/llama.md) |
| bloom_560m    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bloom.md) |
| bloom_7.1b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bloom.md) |
| bloom_65b     | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bloom.md) |
| bloom_176b    | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/bloom.md) |

### 强化学习

| Algorithm | Discrete Action Space | Continuous Action Space | CPU | GPU | Ascend | Environment | Reward |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [DQN](https://github.com/mindspore-lab/mindrl/tree/master/example/dqn)  | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) | 195 |
| [PPO](https://github.com/mindspore-lab/mindrl/tree/master/example/ppo)  | / | ✅ | ✅ | ✅ | ✅ | [HalfCheetah-v2](https://www.gymlibrary.dev/environments/mujoco/half_cheetah/) | 4800 |
| [AC](https://github.com/mindspore-lab/mindrl/tree/master/example/ac) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) | 195 |
| [A2C](https://github.com/mindspore-lab/mindrl/tree/master/example/a2c) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) | 195 |
| [DDPG](https://github.com/mindspore-lab/mindrl/tree/master/example/ddpg)  | / | ✅ | ✅ | ✅ | ✅ | [HalfCheetah-v2](https://www.gymlibrary.dev/environments/mujoco/half_cheetah/) | 4800 |
| [QMIX](https://github.com/mindspore-lab/mindrl/tree/master/example/qmix)  | ✅ | / | ✅ | ✅ | ✅ | [SMAC/Simple Spread](https://github.com/oxwhirl/smac/) | 90%/-145 |
| [SAC](https://github.com/mindspore-lab/mindrl/tree/master/example/sac)  | / | ✅ | ✅ | ✅ | ✅ | [HalfCheetah-v2](https://www.gymlibrary.dev/environments/mujoco/half_cheetah/) | 4800 |
| [TD3](https://github.com/mindspore-lab/mindrl/tree/master/example/td3)  | / | ✅ | ✅ | ✅ | ✅ | [HalfCheetah-v2](https://www.gymlibrary.dev/environments/mujoco/half_cheetah/) | 4800 |
| [C51](https://github.com/mindspore-lab/mindrl/tree/master/example/c51) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) | 195 |
| [A3C](https://github.com/mindspore-lab/mindrl/tree/master/example/a3c) | ✅ | / | / | ✅ | ✅ | [CartPole-v0](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) | 195 |
| [CQL](https://github.com/mindspore-lab/mindrl/tree/master/example/cql) | / | ✅ | ✅ | ✅ | ✅ | [Hopper-v0](https://www.gymlibrary.dev/environments/mujoco/hopper) | 3500 |
| [MAPPO](https://github.com/mindspore-lab/mindrl/tree/master/example/mappo)  | ✅ | / | ✅ | ✅ | ✅ | [Simple Spread](https://github.com/openai/multiagent-particle-envs) | -145 |
| [GAIL](https://github.com/mindspore-lab/mindrl/tree/master/example/gail) | / | ✅ | ✅ | ✅ | ✅ | [HalfCheetah-v2](https://www.gymlibrary.dev/environments/mujoco/half_cheetah/) | 4800 |
| [AWAC](https://github.com/mindspore-lab/mindrl/tree/master/example/awac) | / | ✅ | ✅ | ✅ | ✅ | [Ant-v2](https://www.gymlibrary.dev/environments/mujoco/ant) | 5000 |
| [Dreamer](https://github.com/mindspore-lab/mindrl/tree/master/example/dreamer)  | / | ✅ | / | ✅ | ✅ | [Walker-walk](https://github.com/deepmind/dm_control) | 900 |
| [IQL](https://github.com/mindspore-lab/mindrl/tree/master/example/iql) | / | ✅ | ✅ | ✅ | ✅ | [Walker2d-v2](https://www.gymlibrary.dev/environments/mujoco/walker2d/) | 3000 |
| [MADDPG](https://github.com/mindspore-lab/mindrl/tree/master/example/maddpg)  | ✅ | / | ✅ | ✅ | ✅ | [simple_spread](https://pettingzoo.farama.org/environments/mpe/simple_spread/) | -140 |
| [Double DQN](https://github.com/mindspore-lab/mindrl/tree/master/example/double_dqn) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) | 195 |
| [Policy Gradient](https://github.com/mindspore-lab/mindrl/tree/master/example/pg) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) | 195 |
| [Dueling DQN](https://github.com/mindspore-lab/mindrl/tree/master/example/dueling_dqn) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) | 195 |

## 科学计算套件

| 领域   | 网络                                                                                                                                               |                                                                   MindSpore实现                                                                    | Ascend | GPU |
|------|--------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:---:|
| 通用物理 | [deepbsde](https://www.pnas.org/doi/10.1073/pnas.1718942115)                                                                                     |                             [Link](https://gitee.com/mindspore/models/blob/r2.1/research/hpc/deepbsde/README.md#)                              |        |  ✅  |
| 通用物理 | [pfnn](https://www.sciencedirect.com/science/article/abs/pii/S0021999120308597)                                                                  |                              [Link](https://gitee.com/mindspore/models/blob/r2.1/research/hpc/pfnn/README_CN.md#)                              |        |  ✅  |
| 通用物理 | [pinns](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)                                                                 |                             [Link](https://gitee.com/mindspore/models/blob/r2.1/research/hpc/pinns/README_CN.md#)                              |        |  ✅  |
| 海洋物理 | [ocean_model](https://gmd.copernicus.org/articles/12/4729/2019/)                                                                                 |                            [Link](https://gitee.com/mindspore/models/blob/r2.1/research/hpc/ocean_model/README.md#)                            |        |  ✅  |
| 电磁学  | [incremental_learning](https://arxiv.org/abs/2111.08823)                                                                                         |         [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindElec/examples/physics_driven/incremental_learning/README_CN.md#)          |   ✅    |  ✅  |
| 电磁学  | [pinn_fwi](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021JB023120)                                                                 |                      |   ✅    |  ✅  |
| 电磁学  | [time_domain_maxwell](https://www.ijcai.org/proceedings/2022/533)                                                                                |          [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindElec/examples/physics_driven/time_domain_maxwell/README_CN.md#)          |   ✅    |  ✅  |
| 电磁学  | [frequency_domain_maxwell](https://arxiv.org/abs/2107.06164)                                                                                     |       [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindElec/examples/physics_driven/frequency_domain_maxwell/README_CN.md#)        |   ✅    |  ✅  |
| 电磁学  | [AD_FDTD](https://www.mindspore.cn/mindelec/docs/zh-CN/r0.2/AD_FDTD.html)                                                                        |                       [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindElec/examples/AD_FDTD/README_CN.md#)                        |   ✅    |  ✅  |
| 电磁学  | [SED_ANN](https://gitee.com/mindspore/mindscience/tree/r0.3/MindElec/examples/data_driven/sed_ann)                                             |                 [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindElec/examples/data_driven/sed_ann/README_CN.md#)                  |   ✅    |  ✅  |
| 电磁学  | [Metasurface_holograms](https://www.researching.cn/articles/OJ44d3746c3db8c1e1)                                                                  |          [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindElec/examples/metasurface/metasurface_holograms/README_CN.md#)           |   ✅    |  ✅  |
| 计算生物 | [MEGA-Fold](https://arxiv.org/abs/2206.12240v1)                                                                                                  |                  [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/MEGAProtein/README_CN.md#)                   |   ✅    |  ✅  |
| 计算生物 | [MEGA-EvoGen](https://arxiv.org/abs/2208.09652)                                                                                                  |                  [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/MEGAProtein/README_CN.md#)                   |   ✅    |  ✅  |
| 计算生物 | [MEGA-Assessment](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/MEGAProtein/README_CN.md#)                         |                  [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/MEGAProtein/README_CN.md#)                   |   ✅    |  ✅  |
| 计算生物 | [ColabDesign](https://www.biorxiv.org/content/10.1101/2021.11.10.468128.abstract)                                                                |                     [Link](https://gitee.com/mindspore/mindscience/tree/r0.3/MindSPONGE/applications/research/Colabdesign)                     |   ✅    |  ✅  |
| 计算生物 | [DeepDR](https://academic.oup.com/bioinformatics/article/35/24/5191/5497253)                                                                     |                  [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/DeepDR/README.md#)                  |   ✅    |  ✅  |
| 计算生物 | [DeepFRI](https://www.nature.com/articles/s41467-021-23303-9)                                                                                    |                [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/DeepFRI/README_CN.md#)                |   ✅    |  ✅  |
| 计算生物 | [FAAST](https://www.biorxiv.org/content/10.1101/2023.04.14.536890v1)                                                                             |                 [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/FAAST/README_CN.md#)                 |   ✅    |  ✅  |
| 计算生物 | [JT-VAE](https://www.biorxiv.org/content/10.1101/2023.04.14.536890v1)                                                                            |                  [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/JT-VAE/README.md#)                  |   ✅    |  ✅  |
| 计算生物 | [KGNN](https://www.ijcai.org/Proceedings/2020/0380.pdf)                                                                                          |                   [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/KGNN/README.md#)                   |   ✅    |  ✅  |
| 计算生物 | [MG-BERT](https://academic.oup.com/bib/article-abstract/22/6/bbab152/6265201)                                                                    |                 [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/MG_BERT/README.md#)                  |   ✅    |  ✅  |
| 计算生物 | [Multimer](https://www.biorxiv.org/content/10.1101/2021.10.04.463034v1)                                                                          |                 [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/Multimer/README.md#)                 |   ✅    |  ✅  |
| 计算生物 | [ProteinMPNN](https://www.science.org/doi/abs/10.1126/science.add2187)                                                                           |               [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/ProteinMPNN/README.md#)                |   ✅    |  ✅  |
| 计算生物 | [UFold](https://doi.org/10.1093/nar/gkab1074)                                                                                                    |                 [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/UFold/README_CN.md#)                 |   ✅    |  ✅  |
| 计算生物 | [ESM-IF1](https://proceedings.mlr.press/v162/hsu22a.html)                                                                                        |                  [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/esm/README_CN.md#)                  |   ✅    |  ✅  |
| 计算生物 | [ESM2](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf)                                                                     |                  [Link](https://gitee.com/mindspore/mindscience/tree/r0.3/MindSPONGE/mindsponge/python/pipeline/models/esm2)                   |   ✅    |  ✅  |
| 计算生物 | [Grover](https://proceedings.neurips.cc/paper/2020/file/94aef38441efa3380a3bed3faf1f9d5d-Paper.pdf)                                              |                  [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/grover/README.md#)                  |   ✅    |  ✅  |
| 计算生物 | [Pafnucy](https://doi.org/10.1093/bioinformatics/bty374)                                                                                         |                 [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/pafnucy/README.md#)                  |   ✅    |  ✅  |
| 计算生物 | [SchNet](https://arxiv.org/abs/1706.08566)                                                                                                       |                                 [Link](https://gitee.com/mindspore/mindscience/tree/r0.3/MindSPONGE/cybertron)                                 |   ✅    |  ✅  |
| 计算生物 | [MolCT](https://arxiv.org/abs/2012.11816)                                                                                                        |                                 [Link](https://gitee.com/mindspore/mindscience/tree/r0.3/MindSPONGE/cybertron)                                 |   ✅    |  ✅  |
| 计算生物 | [PhysNet](https://arxiv.org/abs/1902.08408)                                                                                                      |                                 [Link](https://gitee.com/mindspore/mindscience/tree/r0.3/MindSPONGE/cybertron)                                 |   ✅    |  ✅  |
| 计算流体 | [FNO1D](https://arxiv.org/abs/2010.08895)                                                                                                        |             [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/burgers_fno/FNO1D_CN.ipynb)             |   ✅    |  ✅  |
| 计算流体 | [KNO1D](https://arxiv.org/abs/2301.10022)                                                                                                        |             [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/burgers_kno/README_CN.md#)              |   ✅    |  ✅  |
| 计算流体 | [FNO2D](https://arxiv.org/abs/2010.08895)                                                                                                        |          [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/navier_stokes_fno/FNO2D_CN.ipynb)          |   ✅    |  ✅  |
| 计算流体 | [KNO2D](https://arxiv.org/abs/2301.10022)                                                                                                        |          [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/navier_stokes_kno/README_CN.md#)           |   ✅    |  ✅  |
| 计算流体 | [FNO3D](https://arxiv.org/abs/2010.08895)                                                                                                        |        [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/navier_stokes_3d_fno/FNO3D_CN.ipynb)         |   ✅    |  ✅  |
| 计算流体 | [CAE-LSTM](https://doi.org/10.13700/j.bh.1001-5965.2022.0085)                                                                                    |     |   ✅    |  ✅  |
| 计算流体 | [eHDNN](https://doi.org/10.1016/j.ast.2022.107636)                                                                                               |        [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/transonic_buffet_ehdnn/README_CN.md#)        |   ✅    |  ✅  |
| 计算流体 | [HDNN](https://doi.org/10.1016/j.ast.2022.107636)                                                                                                |          [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/move_boundary_hdnn/README_CN.md#)          |   ✅    |  ✅  |
| 计算流体 | [ViT](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/airfoil/2D_steady/2D_steady_CN.ipynb)                |          [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/airfoil/2D_steady/README_CN.md#)           |   ✅    |  ✅  |
| 计算流体 | [PeRCNN](https://www.nature.com/articles/s42256-023-00685-7)                                                                                     |        |   ✅    |  ✅  |
| 计算流体 | [Burgers1D](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)                                                             |              [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/physics_driven/burgers/README_CN.md#)              |   ✅    |  ✅  |
| 计算流体 | [Cylinder Flow](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/physics_driven/cylinder_flow/navier_stokes2D_CN.ipynb) |           [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/physics_driven/cylinder_flow/README_CN.md#)           |   ✅    |  ✅  |
| 计算流体 | [PDE-Net](https://arxiv.org/abs/1710.09668)                                                                                                      | [Link](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_mechanism_fusion/variant_linear_coe_pde_net/README_CN.md#) |   ✅    |  ✅  |

## 大模型套件

### Transformers

| model                            | dataset     | metric                               | score                       | mindformers config                                                                                                               |
|----------------------------------|-------------|--------------------------------------|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| bert_base_uncased                | wiki        | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/bert/run_bert_base_uncased.yaml)                                         |
| bert_tiny_uncased                | wiki        | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/bert/run_bert_tiny_uncased.yaml)                                         |
| qa_bert_base_uncased             | SQuAD v1.1  | EM / F1                              | 80.74 / 88.33               | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/qa/run_qa_bert_base_uncased.yaml)                                        |
| tokcls_bert_base_chinese_cluener | CLUENER     | Entity F1                            | 0.7905                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/tokcls/run_tokcls_bert_base_chinese_cluener.yaml)                        |
| txtcls_bert_base_uncased_mnli    | Mnli        | Entity F1                            | 84.80%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/txtcls/run_txtcls_bert_base_uncased_mnli.yaml)                           |
| clip_vit_b_32                    | Cifar100    | Accuracy                             | 57.24%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml)     |
| clip_vit_b_16                    | Cifar100    | Accuracy                             | 61.41%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_b_16_zero_shot_image_classification_cifar100.yaml)     |
| clip_vit_l_14                    | Cifar100    | Accuracy                             | 69.67%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_l_14_zero_shot_image_classification_cifar100.yaml)     |
| clip_vit_l_14@336                | Cifar100    | Accuracy                             | 68.19%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_l_14@336_zero_shot_image_classification_cifar100.yaml) |
| filip_vit_l_14                   | - | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/filip/run_filip_vit_l_14.yaml)                                           |
| glm_6b                           | ADGEN       | BLEU-4 / Rouge-1 / Rouge-2 / Rouge-l | 8.42 / 31.75 / 7.98 / 25.28 | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/glm/run_glm_6b_infer.yaml)                                               |
| gpt2                             | wikitext-2  | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2.yaml)                                                      |
| gpt2_13b                         | wikitext-2  | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_13b.yaml)                                                  |
| gpt2_52b                         | wikitext-2  | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_52b.yaml)                                                  |
| llama_7b                         | alpac       | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_7b.yaml)                                                 |
| llama_13b                        | alpac       | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_13b.yaml)                                                |
| llama_65b                        | alpac       | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_65b.yaml)                                                |
| mae_vit_base_p16                 | ImageNet-1K | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/mae/run_mae_vit_base_p16_224_800ep.yaml)                                 |
| vit_base_p16                     | ImageNet-1K | Accuracy                             | 83.71%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/vit/run_vit_base_p16_224_100ep.yaml)                                     |
| pangualpha_2_6b                  | 悟道数据集       | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/pangualpha/run_pangualpha_2_6b.yaml)                                     |
| pangualpha_13b                   | 悟道数据集       | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/pangualpha/run_pangualpha_13b.yaml)                                      |
| swin_base_p4w7                   | ImageNet-1K | Accuracy                             | 83.44%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/swin/run_swin_base_p4w7_224_100ep.yaml)                                  |
| t5_small                         | WMT16       | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/t5/run_t5_small_on_wmt16.yaml)                                           |
| t5_tiny                          | WMT16       | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/t5/run_t5_tiny_on_wmt16.yaml)                                            |

### 推荐

| model | dataset | auc | mindrec recipe | vanilla mindspore |
| --- | --- | --- | --- | --- |
| Wide&Deep  | Criteo | 0.8 | [link](https://github.com/mindspore-lab/mindrec/tree/r0.3/models/wide_deep) | [link](https://gitee.com/mindspore/models/tree/r2.1/official/recommend/Wide_and_Deep) |
| Deep&Cross Network (DCN) | Criteo | 0.8 | [link](https://github.com/mindspore-lab/mindrec/tree/r0.3/models/deep_and_cross) | [link](https://gitee.com/mindspore/models/tree/r2.1/research/recommend/deep_and_cross) |
