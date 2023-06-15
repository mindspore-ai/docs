# 官方模型库

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/official_models.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

## 领域套件与扩展包

### 计算机视觉

#### 图像分类（骨干类)

以下数据基于Ascend 910A环境和ImageNet-1K数据集获得。

| model | acc@1 | mindcv recipe | vanilla mindspore |
:-:     | :-:        | :-:    | :-:  
|  vgg11           |  71.86   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |
|  vgg13           |  72.87   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |
|  vgg16           |  74.61   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/VGG/vgg16)
|  vgg19           |  75.21   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vgg)           |   [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/VGG/vgg19)
| resnet18         |  70.21   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet)
| resnet34         |  74.15   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet)
| resnet50         |  76.69   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet)
| resnet101        |  78.24   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet)
| resnet152        |  78.72   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnet)        |   [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ResNet)
| resnetv2_50      |  76.90   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnetv2)      |
| resnetv2_101     |  78.48   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnetv2)      |
| dpn92            |  79.46   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |
| dpn98            |  79.94   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |
| dpn107           |  80.05   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |
| dpn131           |  80.07   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/dpn)           |
| densenet121      |  75.64   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |
| densenet161      |  79.09   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |
| densenet169      |  77.26   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |
| densenet201      |  78.14   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/densenet)      |
| seresnet18       |  71.81   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |
| seresnet34       |  75.36   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |
| seresnet50       |  78.31   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |
| seresnext26      |  77.18   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |
| seresnext50      |  78.71   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/senet)         |
| skresnet18       |  73.09   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet)              |
| skresnet34       |  76.71   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet)              |
| skresnet50_32x4d |  79.08   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/sknet)              |
| resnext50_32x4d  |  78.53   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |
| resnext101_32x4d |  79.83   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |
| resnext101_64x4d |  80.30   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |
| resnext152_64x4d |  80.52   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnext)            |
| rexnet_x09       |  77.07   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |
| rexnet_x10       |  77.38   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |
| rexnet_x13       |  79.06   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |
| rexnet_x15       |  79.94   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |
| rexnet_x20       |  80.64   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/rexnet)             |
| resnest50        |  80.81   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnest)            |
| resnest101       |  82.50   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/resnest)            |
| res2net50        |  79.35   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |
| res2net101       |  79.56   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |
| res2net50_v1b    |  80.32   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |
| res2net101_v1b   |  95.41   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/res2net)            |
| googlenet        |  72.68   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/googlenet)          |
| inceptionv3      |  79.11   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/inceptionv3)        | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/Inception/inceptionv3) |
| inceptionv4      |  80.88   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/inceptionv4)        | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/Inception/inceptionv4) |
| mobilenet_v1_025 |  53.87   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |
| mobilenet_v1_050 |  65.94   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |
| mobilenet_v1_075 |  70.44   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |
| mobilenet_v1_100 |  72.95   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv1)        |
| mobilenet_v2_075 |  69.98   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |
| mobilenet_v2_100 |  72.27   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |
| mobilenet_v2_140 |  75.56   |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv2)        |
| mobilenet_v3_small     | 68.10 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3)      |
| mobilenet_v3_large     | 75.23 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilenetv3)      | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/MobileNet/mobilenetv3) |
| shufflenet_v1_g3_x0_5  | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv1)     |
| shufflenet_v1_g3_x1_5  | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv1)     | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ShuffleNet/shufflenetv1) |
| shufflenet_v2_x0_5     | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |
| shufflenet_v2_x1_0     | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/ShuffleNet/shufflenetv2) |
| shufflenet_v2_x1_5     | 57.05 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |
| shufflenet_v2_x2_0     | 67.77 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/shufflenetv2)     |
| xception               | 79.01 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/xception)         | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/Inception/xception) |
| ghostnet_50            | 66.03 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |
| ghostnet_100           | 73.78 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |
| ghostnet_130           | 75.50 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/ghostnet)         |
| nasnet_a_4x1056        | 73.65 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/nasnet)           |
| mnasnet_0.5            | 68.07 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |
| mnasnet_0.75           | 71.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |
| mnasnet_1.0            | 74.28 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |
| mnasnet_1.4            | 76.01 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mnasnet)          |
| efficientnet_b0        | 76.89 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/efficientnet)     | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/Efficientnet/efficientnet-b0)
| efficientnet_b1        | 78.95 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/efficientnet)     | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/Efficientnet/efficientnet-b1)
| efficientnet_b2        | 79.80 |     | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/Efficientnet/efficientnet-b2)
| efficientnet_b3        | 80.50 |     | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/Efficientnet/efficientnet-b3)
| efficientnet_v2        | 83.77 |     | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/Efficientnet/efficientnetv2)
| regnet_x_200mf         | 68.74 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |
| regnet_x_400mf         | 73.16 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |
| regnet_x_600mf         | 73.34 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |
| regnet_x_800mf         | 76.04 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |
| regnet_y_200mf         | 70.30 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |
| regnet_y_400mf         | 73.91 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |
| regnet_y_600mf         | 75.69 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |
| regnet_y_800mf         | 76.52 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/regnet)     |
| mixnet_s               | 75.52 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet)     |
| mixnet_m               | 76.64 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet)     |
| mixnet_l               | 78.73 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mixnet)     |
| hrnet_w32              | 80.64 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/hrnet)      |
| hrnet_w48              | 81.19 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/hrnet)      |
| bit_resnet50           | 76.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit)         |
| bit_resnet50x3         | 80.63 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit)        |
| bit_resnet101          | 77.93 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/bit)        |
| repvgg_a0              | 72.19 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |
| repvgg_a1              | 74.19 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |
| repvgg_a2              | 76.63 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |
| repvgg_b0              | 74.99 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |
| repvgg_b1              | 78.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |
| repvgg_b2              | 79.29 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)     |
| repvgg_b3            | 80.46 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)       |
| repvgg_b1g2          | 78.03 |[config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)        |
| repvgg_b1g4          | 77.64 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)       |
| repvgg_b2g4          | 78.80 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repvgg)       |
| repmlp_t224          | 76.71 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/repmlp)       |
| convnext_tiny        | 81.91 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext)     |
| convnext_small       | 83.40 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext)     |
| convnext_base        | 83.32 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convnext)     |
| vit_b_32_224         | 75.86 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)          |  [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/VIT)
| vit_l_16_224         | 76.34| [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)           |
| vit_l_32_224         | 73.71 |  [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/vit)         |
| swintransformer_tiny | 80.82 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/swintransformer) | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/SwinTransformer)
| pvt_tiny            | 74.81 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |
| pvt_small           | 79.66 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |
| pvt_medium          | 81.82 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |
| pvt_large           | 81.75 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvt)           |
| pvt_v2_b0           | 71.50 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |
| pvt_v2_b1           | 78.91 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |
| pvt_v2_b2           | 81.99 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |
| pvt_v2_b3           | 82.84 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |
| pvt_v2_b4           | 83.14 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pvtv2)         |
| pit_ti              | 72.96 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |
| pit_xs              | 78.41 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |
| pit_s               | 80.56 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |
| pit_b               | 81.87 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/pit)           |
| coat_lite_tiny      | 77.35 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat)          |
| coat_lite_mini      | 78.51 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat)          |
| coat_tiny           | 79.67 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/coat)          |
| convit_tiny         | 73.66 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |
| convit_tiny_plus    | 77.00 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |
| convit_small        | 81.63 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |
| convit_small_plus   | 81.80 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |
| convit_base         | 82.10 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |
| convit_base_plus    | 81.96 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/convit)        |
| crossvit_9          | 73.56 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit)      |
| crossvit_15         | 81.08 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit)      |
| crossvit_18         | 81.93 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/crossvit)      |
| mobilevit_xx_small  | 68.90 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit)     |
| mobilevit_x_small   | 74.98 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit)     |
| mobilevit_small     | 78.48 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/mobilevit)     |
| visformer_tiny      | 78.28 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |
| visformer_tiny_v2   | 78.82 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |
| visformer_small     | 81.76 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |
| visformer_small_v2  | 82.17 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/visformer)     |
| edgenext_xx_small   | 71.02 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |
| edgenext_x_small    | 75.14 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |
| edgenext_small      | 79.15 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |
| edgenext_base       | 82.24 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/edgenext)      |
| poolformer_s12      | 77.33 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/poolformer)    |
| xcit_tiny_12_p16    | 77.67 | [config](https://github.com/mindspore-lab/mindcv/tree/main/configs/xcit)          |

### 目标检测

以下数据基于Ascend 910A环境和COCO2017数据集获得。

#### yolo

| model | map |  mindyolo recipe | vanilla mindspore
:-: | :-: | :-: | :-: |
| yolov8_n <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | 37.2 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov8_s <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | 44.6 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov8_m <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | 50.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov8_l <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | 52.8 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov8_x <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | 53.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov8) |
| yolov7_t <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | 37.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |
| yolov7_l <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | 50.8 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |
| yolov7_x <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> |  52.4| [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov7) |
| yolov5_n | 27.3 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov5_s | 37.6 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv5) |
| yolov5_m | 44.9 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov5_l | 48.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov5_x | 50.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov5) |
| yolov4_csp       | 45.4 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) |
| yolov4_csp(silu) | 45.8 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov4) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv4) |
| yolov3_darknet53 | 45.5 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolov3) | [link](https://gitee.com/mindspore/models/tree/master/official/cv/YOLOv3) |
| yolox_n | 24.1 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_t | 33.3 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_s | 40.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_m | 46.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_l | 49.2 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_x | 51.6 |[config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |
| yolox_darknet53 | 47.7 | [config](https://github.com/mindspore-lab/mindyolo/tree/master/configs/yolox) |

#### 经典

| model |  map | mind_series recipe | vanilla mindspore
:-:     |  :-:            | :-:        |  :-:
|  ssd_vgg16                 | 23.2  |   | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/SSD)|
|  ssd_mobilenetv1           | 22.0  |   | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/SSD)|
|  ssd_mobilenetv2           | 29.1  |   | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/SSD)|
|  ssd_resnet50              | 34.3  |   | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/SSD)|
|  fastrcnn                  | 58    |   | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/FasterRCNN) |
|  maskrcnn_mobilenetv1      | coming soon   |   | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/MaskRCNN/maskrcnn_mobilenetv1) |
|  maskrcnn_resnet50         | coming soon   |   | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/MaskRCNN/maskrcnn_resnet50) |

### 语义分割

| model |  mind_series recipe | vanilla mindspore
:-:     |  :-:            | :-:
| ocrnet          |      | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/OCRNet/)         |
| deeplab v3      |      | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/DeepLabv3)       |
| deeplab v3 plus |      | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/DeepLabV3P)      |
| unet            |      | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/Unet)            |

### OCR

| model | dataset | fscore | mindocr recipe | vanilla mindspore
:-:     |  :-:       | :-:        | :-:   | :-:
| dbnet_mobilenetv3  | icdar2015          | 77.28 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet18     | icdar2015          | 83.71 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet_resnet50     | icdar2015          | 84.99 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/DBNet/)  |
| dbnet_resnet50     | msra-td500         | 85.03 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| dbnet++_resnet50   | icdar2015          | 86.60 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/dbnet)  |
| psenet_resnet152   | icdar2015          | 82.06 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/psenet) | [link](https://gitee.com/mindspore/models/tree/r2.0/research/cv/psenet)   |
| east_resnet50      | icdar2015          | 84.87 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/det/east)   | [link](https://gitee.com/mindspore/models/tree/r2.0/research/cv/east)     |
| svtr_tiny          | IC03,13,15,IIT,etc | 89.02 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/svtr)   |  
| crnn_vgg7          | IC03,13,15,IIT,etc | 82.03 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/CRNN)     |
| crnn_resnet34_vd   | IC03,13,15,IIT,etc | 84.45 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/crnn)   |
| rare_resnet34_vd   | IC03,13,15,IIT,etc | 85.19 | [config](https://github.com/mindspore-lab/mindocr/tree/main/configs/rec/rare)   |

### Face

| model | dataset | acc | mindface recipe | vanilla mindspore
:-:     |  :-:       | :-:        | :-:   | :-:
| arcface_mobilefacenet-0.45g  | MS1MV2          | 98.70  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |
| arcface_r50                  | MS1MV2          | 99.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |
| arcface_r100                 | MS1MV2          | 99.38  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/Arcface) |
| arcface_vit_t                | MS1MV2          | 99.71  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |
| arcface_vit_s                | MS1MV2          | 99.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |
| arcface_vit_b                | MS1MV2          | 99.81  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |
| arcface_vit_l                | MS1MV2          | 99.75  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/recognition)  |
| retinaface_mobilenet_0.25    | WiderFace        | 90.77/88.2/74.76  | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/detection)  | [link](https://gitee.com/mindspore/models/tree/r2.0/research/cv/retinaface) |
| retinaface_r50               | WiderFace        | 95.07/93.61/84.84 | [config](https://github.com/mindspore-lab/mindface/tree/main/mindface/detection)  | [link](https://gitee.com/mindspore/models/tree/r2.0/official/cv/RetinaFace_ResNet50) |

### 强化学习

| Algorithm | RL Version | Action Space (Discrete) | Action Space (Continuous) | Device (CPU) | Device (GPU) | Device (Ascend) | Example Environment |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [DQN](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fdqn) <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | >= 0.1 | ✔️ | / | ✔️ | ✔️ | ✔️| [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [PPO](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fppo) <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | >= 0.1 | / | ✔️ | ✔️ | ✔️ | ✔️ | [HalfCheetah-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhalf_cheetah%2F) |
| [AC](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fac) | >= 0.1 | ✔️ | / | ✔️ | ✔️ | ✔️ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [A2C](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fa2c) | >= 0.2 | ✔️ | / | ✔️ | ✔️ | ✔️ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [DDPG](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fddpg) <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | >= 0.3 | / | ✔️ | ✔️ | ✔️ | ✔️ | [HalfCheetah-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhalf_cheetah%2F) |
| [QMIX](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fqmix) <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | >= 0.5 | ✔️ | / | ✔️ | ✔️ | ✔️ | [SMAC, Simple Spread](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Foxwhirl%2Fsmac%2F) |
| [SAC](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fsac) <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | >= 0.5 | / | ✔️ | ✔️ | ✔️ | ✔️ | [HalfCheetah-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhalf_cheetah%2F) |
| [TD3](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Ftd3) <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | >= 0.6 | / | ✔️ | ✔️ | ✔️ | ✔️ | [HalfCheetah-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhalf_cheetah%2F) |
| [C51](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fc51) | >= 0.6 | ✔️ | / | ✔️ | ✔️ | ✔️ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [A3C](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fa3c) | >= 0.6 | ✔️ | / | / | ✔️ | ✔️ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [CQL](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fcql) | >= 0.6 | / | ✔️ | ✔️ | ✔️ | ✔️ | [Hopper-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhopper) |
| [MAPPO](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fmappo) <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | >= 0.6 | ✔️ | / | ✔️ | ✔️ | ✔️ | [Simple Spread](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fopenai%2Fmultiagent-particle-envs) |
| [GAIL](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fgail) | >= 0.6 | / | ✔️ | ✔️ | ✔️ | ✔️ | [HalfCheetah-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fhalf_cheetah%2F) |
| [MCTS](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fmcts) | >= 0.6 | ✔️ | / | ✔️ | ✔️ | / | [Tic-Tac-Toe](https://gitee.com/mindspore/reinforcement/blob/master/mindspore_rl/environment/tic_tac_toe_environment.py) |
| [AWAC](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fawac) | >= 0.6 | / | ✔️ | ✔️ | ✔️ | ✔️ | [Ant-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fant) |
| [Dreamer](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fdreamer) <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | >= 0.6 | / | ✔️ | / | ✔️ | ✔️️️️ | [Walker-walk](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fdeepmind%2Fdm_control) |
| [IQL](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fiql) | >= 0.6 | / | ✔️ | ✔️ | ✔️ | ✔️ | [Walker2d-v2](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fmujoco%2Fwalker2d%2F) |
| [MADDPG](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fmaddpg) <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | >= 0.6 | ✔️ | / | ✔️ | ✔️ | ✔️ | [simple_spread](https://gitee.com/link?target=https%3A%2F%2Fpettingzoo.farama.org%2Fenvironments%2Fmpe%2Fsimple_spread%2F) |
| [Double DQN](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fdouble_dqn) | >= 0.6 | ✔️ | / | ✔️ | ✔️ | ✔️ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [Policy Gradient](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fpg) | >= 0.6 | ✔️ | / | ✔️ | ✔️ | ✔️ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |
| [Dueling DQN](https://gitee.com/link?target=https%3A%2F%2Fgithub.com%2Fmindspore-lab%2Fmindrl%2Ftree%2Fmaster%2Fexample%2Fdueling_dqn) | >= 0.6 | ✔️ | / | ✔️ | ✔️ | ✔️ | [CartPole-v0](https://gitee.com/link?target=https%3A%2F%2Fwww.gymlibrary.dev%2Fenvironments%2Fclassic_control%2Fcart_pole%2F) |

## 科学计算套件

## 大模型套件

### Transformers

| model_name | model                            | dataset     | metric                               | score                       | mindformers config                                                                                                               |
|------------|----------------------------------|-------------|--------------------------------------|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| bert       | bert_base_uncased                | wiki        |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/bert/run_bert_base_uncased.yaml)                                         |
|            | bert_tiny_uncased                | wiki        |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/bert/run_bert_tiny_uncased.yaml)                                         |
|            | qa_bert_base_uncased             | SQuAD v1.1  | EM / F1                              | 80.74 / 88.33               | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/qa/run_qa_bert_base_uncased.yaml)                                        |
|            | tokcls_bert_base_chinese_cluener | CLUENER     | Entity F1                            | 0.7905                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/tokcls/run_tokcls_bert_base_chinese_cluener.yaml)                        |
|            | txtcls_bert_base_uncased_mnli    | Mnli        | Entity F1                            | 84.80%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/txtcls/run_txtcls_bert_base_uncased_mnli.yaml)                           |
| clip       | clip_vit_b_32                    | Cifar100    | Accuracy                             | 57.24%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_b_32_zero_shot_image_classification_cifar100.yaml)     |
|            | clip_vit_b_16                    | Cifar100    | Accuracy                             | 61.41%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_b_16_zero_shot_image_classification_cifar100.yaml)     |
|            | clip_vit_l_14                    | Cifar100    | Accuracy                             | 69.67%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_l_14_zero_shot_image_classification_cifar100.yaml)     |
|            | clip_vit_l_14@336                | Cifar100    | Accuracy                             | 68.19%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/clip/run_clip_vit_l_14@336_zero_shot_image_classification_cifar100.yaml) |
| filip      | filip_vit_l_14                   |             |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/filip/run_filip_vit_l_14.yaml)                                           |
| glm <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png">        | glm_6b                           | ADGEN       | BLEU-4 / Rouge-1 / Rouge-2 / Rouge-l | 8.42 / 31.75 / 7.98 / 25.28 | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/glm/run_glm_6b_infer.yaml)                                               |
| gpt2       | gpt2                             | wikitext-2  |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2.yaml)                                                      |
|            | gpt2_13b                         | wikitext-2  |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_13b.yaml)                                                  |
|            | gpt2_52b                         | wikitext-2  |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/gpt2/run_gpt2_52b.yaml)                                                  |
| llama <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png">     | llama_7b                         | alpac       |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_7b.yaml)                                                 |
|            | llama_13b                        | alpac       |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_13b.yaml)                                                |
|            | llama_65b                        | alpac       |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/llama/run_llama_65b.yaml)                                                |
| mae <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png">        | mae_vit_base_p16                 | ImageNet-1K |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/mae/run_mae_vit_base_p16_224_800ep.yaml)                                 |
| vit <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png">       | vit_base_p16                     | ImageNet-1K | Accuracy                             | 83.71%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/vit/run_vit_base_p16_224_100ep.yaml)                                     |
| pangualpha <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | pangualpha_2_6b                  | 悟道数据集       |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/pangualpha/run_pangualpha_2_6b.yaml)                                     |
|            | pangualpha_13b                   | 悟道数据集       |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/pangualpha/run_pangualpha_13b.yaml)                                      |
| swin <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png">      | swin_base_p4w7                   | ImageNet-1K | Accuracy                             | 83.44%                      | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/swin/run_swin_base_p4w7_224_100ep.yaml)                                  |
| t5         | t5_small                         | WMT16       |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/t5/run_t5_small_on_wmt16.yaml)                                           |
|            | t5_tiny                          | WMT16       |                                      |                             | [config](https://gitee.com/mindspore/mindformers/blob/dev/configs/t5/run_t5_tiny_on_wmt16.yaml)                                            |

### 推荐

| model | dataset | auc | mindrec recipe | vanilla mindspore |
| --- | --- | --- | --- | --- |
| Wide&Deep <img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/hot.png"> | Criteo | 0.8 | [config](https://github.com/mindspore-lab/mindrec/tree/master/models/wide_deep) | [link](https://gitee.com/mindspore/models/tree/master/official/recommend/Wide_and_Deep) |
| Deep&Cross Network (DCN) | Criteo | 0.8 | [config](https://github.com/mindspore-lab/mindrec/tree/master/models/deep_and_cross) | [link](https://gitee.com/mindspore/models/tree/master/research/recommend/deep_and_cross) |
