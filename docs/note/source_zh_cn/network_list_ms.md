# MindSpore网络支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `中级` `高级`

<!-- TOC -->

- [MindSpore网络支持](#mindspore网络支持)
    - [Model Zoo](#model-zoo)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/note/source_zh_cn/network_list_ms.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

## Model Zoo

### 标准网络

|  领域 | 子领域  | 网络   | Ascend（Graph） | Ascend（PyNative） | GPU（Graph） | GPU（PyNative） | CPU（Graph） | CPU（PyNative）
|:----  |:-------  |:----   |:----    |:----    |:---- |:---- |:---- |:----
|计算机视觉（CV） | 图像分类（Image Classification）  | [AlexNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/alexnet)   |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [CNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/cnn_direction_model)  |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DenseNet100](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/densenet) |  Doing |  Doing |  Doing |  Doing | Supported | Supported
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DenseNet121](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/densenet) |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DPN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/dpn) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [EfficientNet-B0](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/efficientnet) |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [GoogLeNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/googlenet)   |  Supported     |  Supported | Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [InceptionV3](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/inceptionv3) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [InceptionV4](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/inceptionv4) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet)    |  Supported |  Supported |  Supported |  Supported | Supported | Supported
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet（量化）](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/lenet_quant)    |  Supported |  Doing |  Supported |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV1](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv1)        |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV2](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv2)        |  Supported |  Supported |  Supported |  Supported | Supported | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV2（量化）](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv2_quant)        |  Supported |  Doing |  Supported |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV3](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/mobilenetv3)        |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [NASNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/nasnet) |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-18](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet)   |  Supported |  Supported |  Supported |  Supported | Supported | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50（量化）](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet50_quant)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-101](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet)        |  Supported |  Supported | Supported |  Supported | Doing | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNeXt50](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnext50)    |  Supported |  Doing | Supported |  Supported | Doing | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [SE-ResNet50](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/resnet)       |  Supported | Supported | Doing | Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ShuffleNetV1](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/shufflenetv1)        |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ShuffleNetV2](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/shufflenetv2) |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  |[SqueezeNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/squeezenet) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [Tiny-DarkNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/tinydarknet)  |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [VGG16](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/vgg16)  |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [Xception](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/xception) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [CenterFace](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/centerface)     |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [CTPN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ctpn)     |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Faster R-CNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/faster_rcnn)  |  Supported |  Doing |  Supported |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Mask R-CNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/maskrcnn)  |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  |[Mask R-CNN (MobileNetV1)](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/maskrcnn_mobilenetv1)         |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [RetinaFace-ResNet50](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/retinaface_resnet50)   |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection  | [SSD](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd)                   |  Supported |  Doing | Supported | Supported | Supported | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-MobileNetV1-FPN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd)         |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-Resnet50-FPN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd)                   |  Supported |  Doing | Doing | Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-VGG16](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/ssd)                   |  Supported |  Doing | Doing | Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [WarpCTC](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/warpctc)                    |  Supported |  Doing |  Supported |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-ResNet18](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_resnet18)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-DarkNet53](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_darknet53)   |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-DarkNet53（量化）](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov3_darknet53_quant)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  |[YOLOv4](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/yolov4)         |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 文本检测（Text Detection）  | [DeepText](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/deeptext)                |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 文本检测（Text Detection）  | [PSENet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/psenet)                |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 文本识别（Text Recognition）  | [CNN+CTC](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/cnnctc)                |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [DeepLabV3](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/deeplabv3)   |  Supported |  Doing |  Doing |  Doing | Supported | Doing
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [U-Net2D (Medical)](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [U-Net3D (Medical)](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet3d)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [U-Net++](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/unet)                |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 关键点检测（Keypoint Detection）  |[OpenPose](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/openpose)                |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 关键点检测（Keypoint Detection）  |[SimplePoseNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/simple_pose)                |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 光学字符识别（Optical Character Recognition）  |[CRNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/cv/crnn)                |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [BERT](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/bert)  |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [FastText](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/fasttext)    |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [GNMT v2](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/gnmt_v2)    |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [GRU](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/gru)            |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [MASS](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/mass)    |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [SentimentNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/lstm)    |  Supported |  Doing |  Supported |  Supported | Supported | Supported
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [Transformer](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/transformer)  |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TinyBERT](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/tinybert)   |  Supported |  Supported |  Supported | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TextCNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/nlp/textcnn)            |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction）  | [DeepFM](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/deepfm)    |  Supported |  Supported |  Supported | Supported| Supported | Doing
| 推荐（Recommender） | 推荐系统、搜索、排序（Recommender System, Search, Ranking）  | [Wide&Deep](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/wide_and_deep)      |  Supported |  Supported |  Supported | Supported | Doing | Doing
| 推荐（Recommender） | 推荐系统（Recommender System）  | [NAML](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/naml)             |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 推荐（Recommender） | 推荐系统（Recommender System）  | [NCF](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/recommend/ncf)    |  Supported |  Doing |  Supported | Doing| Doing | Doing
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GCN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/gcn)  |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GAT](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/gat) |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 图神经网络（GNN） | 推荐系统（Recommender System） | [BGCF](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/official/gnn/bgcf) |  Supported |  Doing |  Doing |  Doing | Doing | Doing

### 研究网络

|  领域 | 子领域  | 网络   | Ascend（Graph） | Ascend（PyNative） | GPU（Graph） | GPU（PyNative） | CPU（Graph） | CPU（PyNative）
|:----  |:-------  |:----   |:----    |:----    |:---- |:---- |:---- |:----
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceAttributes](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/FaceAttribute)     |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [FaceDetection](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/FaceDetection)  |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceQualityAssessment](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/FaceQualityAssessment)     |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceRecognition](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/FaceRecognition)     |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceRecognitionForTracking](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/FaceRecognitionForTracking)     |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-GhostNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/ssd_ghostnet)           |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 关键点检测（Key Point Detection）  | [CenterNet](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/centernet)          |  Supported |  Doing | Doing |  Doing | Supported | Doing
| 计算机视觉（CV）  | 图像风格迁移（Image Style Transfer）  | [CycleGAN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/cv/cycle_gan)       |  Doing     |  Doing | Doing |  Supported | Supported | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [DS-CNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/nlp/dscnn)          |  Supported |  Supported |  Doing | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TextRCNN](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/nlp/textrcnn)    |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TPRR](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/nlp/tprr)  |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction） | [AutoDis](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/recommend/autodis)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
|语音（Audio） | 音频标注（Audio Tagging）  | [FCN-4](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/audio/fcn-4)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
|高性能计算（HPC） | 分子动力学（Molecular Dynamics）  |  [DeepPotentialH2O](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/hpc/molecular_dynamics)   |  Supported | Supported|  Doing |  Doing | Doing | Doing
|高性能计算（HPC） | 海洋模型（Ocean Model）  |  [GOMO](https://gitee.com/mindspore/mindspore/tree/master/model_zoo/research/hpc/ocean_model)   |  Doing |  Doing |  Supported |  Doing | Doing | Doing

> 你也可以使用 [MindWizard工具](https://gitee.com/mindspore/mindinsight/tree/master/mindinsight/wizard/) 快速生成经典网络脚本。
