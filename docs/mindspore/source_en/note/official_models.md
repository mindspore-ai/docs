# Official Models

<a href="https://gitee.com/mindspore/docs/blob/r2.1/docs/mindspore/source_en/note/official_models.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.1/resource/_static/logo_source_en.png"></a>

## Domain Suite and Extension Packages

### Computer Vision

#### Image Classification（backbone）

The following results are tested on Ascend 910 with ImageNet-1K.

| model | acc@1 | mindcv recipe | vanilla mindspore |
| :-:     | :-:        | :-:    | :-:  |
|  vgg11           |  71.86   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   |
|  vgg13           |  72.87   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   |
|  vgg16           |  74.61   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/VGG/vgg16) |
|  vgg19           |  75.21   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/VGG/vgg19) |
| resnet18         |  70.21   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet)  |
| resnet34         |  74.15   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet)   |
| resnet50         |  76.69   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet)    |
| resnet101        |  78.24   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet)    |
| resnet152        |  78.72   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/master/official/cv/ResNet)    |
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
| inceptionv3      |  79.11   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/inceptionv3)        | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv3) |
| inceptionv4      |  80.88   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/inceptionv4)        | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/inceptionv4) |
| mobilenet_v1_025 |  53.87   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v1_050 |  65.94   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v1_075 |  70.44   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v1_100 |  72.95   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |   |
| mobilenet_v2_075 |  69.98   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |   |
| mobilenet_v2_100 |  72.27   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |   |
| mobilenet_v2_140 |  75.56   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |   |
| mobilenet_v3_small     | 68.10 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3)      |    |
| mobilenet_v3_large     | 75.23 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3)      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/MobileNet/mobilenetv3) |
| shufflenet_v1_g3_x0_5  | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv1)     |    |
| shufflenet_v1_g3_x1_5  | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv1)     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv1) |
| shufflenet_v2_x0_5     | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |    |
| shufflenet_v2_x1_0     | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/ShuffleNet/shufflenetv2) |
| shufflenet_v2_x1_5     | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |    |
| shufflenet_v2_x2_0     | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |    |
| xception               | 79.01 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/xception)         | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Inception/xception) |
| ghostnet_50            | 66.03 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |   |
| ghostnet_100           | 73.78 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |   |
| ghostnet_130           | 75.50 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |   |
| nasnet_a_4x1056        | 73.65 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/nasnet)           |   |
| mnasnet_0.5            | 68.07 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| mnasnet_0.75           | 71.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| mnasnet_1.0            | 74.28 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| mnasnet_1.4            | 76.01 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |   |
| efficientnet_b0        | 76.89 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/efficientnet)     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b0)
| efficientnet_b1        | 78.95 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/efficientnet)     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b1)
| efficientnet_b2        | 79.80 |     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b2) |
| efficientnet_b3        | 80.50 |     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnet-b3) |
| efficientnet_v2        | 83.77 |     | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Efficientnet/efficientnetv2) |
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
| vit_b_32_224         | 75.86 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)          |  [link](https://gitee.com/mindspore/models/tree/master/official/cv/VIT) |
| vit_l_16_224         | 76.34| [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)           |   |
| vit_l_32_224         | 73.71 |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)         |   |
| swintransformer_tiny | 80.82 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/swintransformer) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SwinTransformer) |
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

### Object Detection

The following results are tested on Ascend 910 with COCO2017.

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
| yolov5_s | 37.6 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv5) |
| yolov5_m | 44.9 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |   |
| yolov5_l | 48.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |   |
| yolov5_x | 50.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |   |
| yolov4_csp       | 45.4 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) |   |
| yolov4_csp(silu) | 45.8 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv4) |
| yolov3_darknet53 | 45.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov3) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv3) |
| yolox_n | 24.1 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_t | 33.3 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_s | 40.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_m | 46.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_l | 49.2 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_x | 51.6 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |
| yolox_darknet53 | 47.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |   |

#### Classic

| model |  map | mind_series recipe | vanilla mindspore |
| :-:     |  :-:            | :-:        |  :-:  |
|  ssd_vgg16                 | 23.2  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  ssd_mobilenetv1           | 22.0  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  ssd_mobilenetv2           | 29.1  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  ssd_resnet50              | 34.3  |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/SSD)|
|  fasterrcnn                  | 58    | [link](https://github.com/mindspore-lab/models/tree/master/official/cv/RCNN)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/FasterRCNN) |
|  maskrcnn_mobilenetv1      | coming soon   |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_mobilenetv1) |
|  maskrcnn_resnet50         | coming soon   |   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/MaskRCNN/maskrcnn_resnet50) |

### Semantic Segmentation

| model |  mind_series recipe | vanilla mindspore |
| :-:     |  :-:            | :-:     |
| ocrnet          |   [link](https://github.com/mindspore-lab/models/tree/master/official/cv/OCRNet)   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/OCRNet/)         |
| deeplab v3      |      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabv3)       |
| deeplab v3 plus |      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DeepLabV3P)      |
| unet            |      | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Unet)            |

### OCR

#### Text Detection

| model  |dataset | fscore | mindocr recipe | vanilla mindspore |
:-:     |   :-:       | :-:        | :-:   |  :-:   |
| dbnet_mobilenetv3  | icdar2015          | 77.23 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DBNet/)  |
| dbnet_resnet18     | icdar2015          | 81.73 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DBNet/)  |
| dbnet_resnet50     | icdar2015          | 85.05 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/DBNet/)  |
| dbnet++_resnet50   | icdar2015          | 86.74 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |   |
| psenet_resnet152   | icdar2015          | 82.06 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) | [link](https://gitee.com/mindspore/models/tree/master/research/cv/psenet)  |
| east_resnet50      | icdar2015          | 84.87 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)   | [link](https://gitee.com/mindspore/models/tree/master/research/cv/east)    |
| fcenet_resnet50    | icdar2015          | 84.12 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/fcenet)   |   |

#### Text Recognition

| model | dataset | acc | mindocr recipe | vanilla mindspore |
:-:     |   :-:       | :-:        | :-:   |  :-:   |
| svtr_tiny          | IC03,13,15,IIIT,etc | 89.02 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/svtr)   |   |
| crnn_vgg7          | IC03,13,15,IIIT,etc | 82.03 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   | [link](https://gitee.com/mindspore/models/tree/master/official/cv/CRNN)    |
| crnn_resnet34_vd   | IC03,13,15,IIIT,etc | 84.45 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |   |
| rare_resnet34_vd   | IC03,13,15,IIIT,etc | 85.19 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare)   | [link](https://gitee.com/mindspore/models/tree/master/research/cv/crnn_seq2seq_ocr)  |

#### Text Orientation Classification

| model | dataset | acc | mindocr recipe |
:-:     |   :-:       | :-:        | :-:   |
| mobilenetv3  | RCTW17,MTWI,LSVT | 94.59 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/cls/mobilenetv3)   |

### Face

| model | dataset | acc | mindface recipe | vanilla mindspore
| :-:     |  :-:       | :-:        | :-:   | :-: |
| arcface_mobilefacenet-0.45g  | MS1MV2          | 98.70  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |   |
| arcface_r50                  | MS1MV2          | 99.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| arcface_r100                 | MS1MV2          | 99.38  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/Arcface) |
| arcface_vit_t                | MS1MV2          | 99.71  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |   |
| arcface_vit_s                | MS1MV2          | 99.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| arcface_vit_b                | MS1MV2          | 99.81  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| arcface_vit_l                | MS1MV2          | 99.75  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |    |
| retinaface_mobilenet_0.25    | WiderFace        | 90.77/88.2/74.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/detection)  | [link](https://gitee.com/mindspore/models/tree/master/research/cv/retinaface) |
| retinaface_r50               | WiderFace        | 95.07/93.61/84.84 | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/detection)  | [link](https://gitee.com/mindspore/models/tree/master/official/cv/RetinaFace_ResNet50) |

### Language Models

| model |  mindformer recipe | vanilla mindspore
| :-:     |  :-:   | :-: |
| bert_base   | [config](https://gitee.com/mindspore/mindformers/blob/dev/docs/model_cards/t5.md) | [link](https://gitee.com/mindspore/models/tree/master/official/nlp/Bert) |
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

### Reinforcement Learning

| Algorithm | Discrete Action Space | Continuous Action Space | CPU | GPU | Ascend | Environment |
| --- | --- | --- | --- | --- | --- | --- |
| [DQN](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fdqn)  | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [PPO](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fppo)  | / | ✅ | ✅ | ✅ | ✅ | [HalfCheetah-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhalf_cheetah%2F) |
| [AC](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fac) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [A2C](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fa2c) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [DDPG](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fddpg)  | / | ✅ | ✅ | ✅ | ✅ | [HalfCheetah-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhalf_cheetah%2F) |
| [QMIX](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fqmix)  | ✅ | / | ✅ | ✅ | ✅ | [SMAC, Simple Spread](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Foxwhirl%2Fsmac%2F) |
| [SAC](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fsac)  | / | ✅ | ✅ | ✅ | ✅ | [HalfCheetah-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhalf_cheetah%2F) |
| [TD3](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Ftd3)  | / | ✅ | ✅ | ✅ | ✅ | [HalfCheetah-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhalf_cheetah%2F) |
| [C51](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fc51) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [A3C](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fa3c) | ✅ | / | / | ✅ | ✅ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [CQL](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fcql) | / | ✅ | ✅ | ✅ | ✅ | [Hopper-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhopper) |
| [MAPPO](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fmappo)  | ✅ | / | ✅ | ✅ | ✅ | [Simple Spread](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fopenai%2Fmultiagent-particle-envs) |
| [GAIL](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fgail) | / | ✅ | ✅ | ✅ | ✅ | [HalfCheetah-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhalf_cheetah%2F) |
| [MCTS](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fmcts) | ✅ | / | ✅ | ✅ | / | [Tic-Tac-Toe](https://github.com/mindspore-lab/mindrl/blob/r0.7/mindspore_rl/environment/tic_tac_toe_environment.py) |
| [AWAC](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fawac) | / | ✅ | ✅ | ✅ | ✅ | [Ant-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fant) |
| [Dreamer](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fdreamer)  | / | ✅ | / | ✅ | ✅ | [Walker-walk](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fdeepmind%2Fdm_control) |
| [IQL](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fiql) | / | ✅ | ✅ | ✅ | ✅ | [Walker2d-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fwalker2d%2F) |
| [MADDPG](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fmaddpg)  | ✅ | / | ✅ | ✅ | ✅ | [simple_spread](https://gitee.com/link?target=https%3A%2F%2Fpettingzoo.farama.org%2Fenvironments%2Fmpe%2Fsimple_spread%2F) |
| [Double DQN](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fdouble_dqn) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [Policy Gradient](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fpg) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [Dueling DQN](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fdueling_dqn) | ✅ | / | ✅ | ✅ | ✅ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |

## Scientific Computing

| Fields    | Network     | Ascend | GPU |    Data Types     |
|-------|---------------------------------|:------:|:---:|:-----------:|
| General Physics  | [*Deep Learning for High-Dimensional PDEs (deepbsde)](https://gitee.com/mindspore/models/blob/master/research/hpc/deepbsde/README.md#)   |        |  ✅  |  Float 32   |
| General Physics  | [*Penalty-free Neural Networks (pfnn)](https://gitee.com/mindspore/models/blob/master/research/hpc/pfnn/README_CN.md#)                   |        |  ✅  |  Float 32   |
| General Physics  | [*Physics Informed Neural Networks (pinns)](https://gitee.com/mindspore/models/blob/master/research/hpc/pinns/README_CN.md#)             |        |  ✅  |  Float 32   |
| Marine Physics  | [*Finite Differential Method for PDEs (ocean_model)](https://gitee.com/mindspore/models/blob/master/research/hpc/ocean_model/README.md#) |        |  ✅  |  Float 32   |
| Electromagnetics   | [*PINNs with Incremental Learning (incremental_learning)](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindElec/examples/physics_driven/incremental_learning/README_CN.md#)        |   ✅    |  ✅  | Float 16/32 |
| Electromagnetics   | [*Multi-scale PINNs (pinn_fwi)](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindElec/examples/physics_driven/pinnFWI/README.md#)                                                  |   ✅    |  ✅  | Float 16/32 |
| Electromagnetics   | [*PINNs for Maxwell Equation (time_domain_maxwell)](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindElec/examples/physics_driven/time_domain_maxwell/README_CN.md#)               |   ✅    |  ✅  | Float 16/32 |
| Computational Biology | [*MEGA-Fold](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/MEGAProtein/README_CN.md#)         |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*MEGA-EvoGen](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/MEGAProtein/README_CN.md#)       |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*MEGA-Assessment](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/MEGAProtein/README_CN.md#)   |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*ColabDesign](https://gitee.com/mindspore/mindscience/tree/r0.2.0/MindSPONGE/applications/research/Colabdesign)           |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*DeepDR](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/research/DeepDR/README.md#)           |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*DeepFRI](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/research/DeepFRI/README_CN.md#)      |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*iterative Folding Assisted peak ASsignmenT](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/FAAST/README_CN.md#)          |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*JT-VAE](https://gitee.com/mindspore/mindscience/blob/r0.3/MindSPONGE/applications/research/JT-VAE/README.md#)           |   ✅   | ✅ |  Float 32  |
| Computational Biology | [*Knowledge Graph Neural Network](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/research/KGNN/README.md#)               |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*MG-BERT](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/research/MG_BERT/README.md#)         |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*Multimer](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/research/Multimer/README.md#)       |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*ProteinMPNN](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/research/ProteinMPNN/README.md#) |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*UFold](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/research/UFold/README.md#)          |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*ESM-IF1](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/research/esm/README_EN.md#)              |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*ESM2]()              |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*GROVER](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/research/grover/README.md#)           |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*pafnucy](https://gitee.com/mindspore/mindscience/blob/r0.2.0/MindSPONGE/applications/research/pafnucy/README.md#)         |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*SchNet](https://gitee.com/mindspore/mindscience/tree/r0.2.0/MindSPONGE/cybertron/)              |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*MolCT](https://gitee.com/mindspore/mindscience/tree/r0.2.0/MindSPONGE/cybertron/)              |   ✅   | ✅ | Float 16/32 |
| Computational Biology | [*PhysNet](https://gitee.com/mindspore/mindscience/tree/r0.2.0/MindSPONGE/cybertron/)              |   ✅   | ✅ | Float 16/32 |
| Computational fluid dynamics | [*FNO1D](https://gitee.com/mindspore/mindscience/tree/r0.3/MindFlow/applications/data_driven/burgers_fno)             |   ✅   | ✅ | Float 16/32 |
| Computational fluid dynamics | [*KNO1D](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/burgers_kno/README.md)              |   ✅   | ✅ | Float 16/32 |
| Computational fluid dynamics | [*FNO2D](https://gitee.com/mindspore/mindscience/tree/r0.3/MindFlow/applications/data_driven/navier_stokes_fno)              |   ✅   | ✅ | Float 16/32 |
| Computational fluid dynamics | [*KNO2D](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/navier_stokes_kno/README.md)             |   ✅   | ✅ | Float 16/32 |
| Computational fluid dynamics | [*ViT](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_driven/airfoil/2D_steady/README.MD)             |   ✅   | ✅ | Float 16/32 |
| Computational fluid dynamics | [*FNO3D](https://gitee.com/mindspore/mindscience/tree/r0.3/MindFlow/applications/data_driven/navier_stokes_3d_fno)              |   ✅   | ✅ | Float 16/32 |
| Computational fluid dynamics | [*Burgers1D](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/physics_driven/cylinder_flow/README.md)            |   ✅   | ✅ | Float 16/32 |
| Computational fluid dynamics | [*NavierStokes](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/physics_driven/cylinder_flow/README.md)              |   ✅   | ✅ | Float 16/32 |
| Computational fluid dynamics | [*PDE-Net](https://gitee.com/mindspore/mindscience/blob/r0.3/MindFlow/applications/data_mechanism_fusion/variant_linear_coe_pde_net/README.md)   |   ✅   | ✅ | Float 16/32 |

## Foundation Model

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
| pangualpha_2_6b                  | WuDaoCorpora       | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/pangualpha/run_pangualpha_2_6b.yaml)                                     |
| pangualpha_13b                   | WuDaoCorpora       | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/pangualpha/run_pangualpha_13b.yaml)                                      |
| swin_base_p4w7                   | ImageNet-1K | Accuracy                             | 83.44%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/swin/run_swin_base_p4w7_224_100ep.yaml)                                  |
| t5_small                         | WMT16       | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/t5/run_t5_small_on_wmt16.yaml)                                           |
| t5_tiny                          | WMT16       | - | - | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/t5/run_t5_tiny_on_wmt16.yaml)                                            |

### Recommender

| model | dataset | auc | mindrec recipe | vanilla mindspore |
| --- | --- | --- | --- | --- |
| Wide&Deep  | Criteo | 0.8 | [link](https://github.com/mindspore-lab/mindrec/tree/r0.3/models/wide_deep) | [link](https://gitee.com/mindspore/models/tree/master/official/recommend/Wide_and_Deep) |
| Deep&Cross Network (DCN) | Criteo | 0.8 | [link](https://github.com/mindspore-lab/mindrec/tree/r0.3/models/deep_and_cross) | [link](https://gitee.com/mindspore/models/tree/master/research/recommend/deep_and_cross) |
