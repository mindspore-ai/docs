# MindSpore网络支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `中级` `高级`

<!-- TOC -->

- [MindSpore网络支持](#mindspore网络支持)
    - [Model Zoo](#model-zoo)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/note/source_zh_cn/network_list_ms.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## Model Zoo

### 标准网络

|  领域 | 子领域  | 网络  | Ascend (Graph) | Ascend (PyNative) | GPU (Graph) | GPU (PyNative) | CPU (Graph) | CPU (PyNative)|Ascend 310|
|:------   |:------| :-----------  |:------:   |:------:   |:------:  |:------:  |:-----: |:-----:|:-----:|
| 计算机视觉（CV） | 图像分类（Image Classification）  | [AlexNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/alexnet)          |   |   |  ✅ |  ✅ |   |   |   |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [CNN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/cnn_direction_model)  |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DenseNet100](https://gitee.com/mindspore/models/tree/r1.3/official/cv/densenet) |    |    |    |    | ✅ | ✅ |   |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DenseNet121](https://gitee.com/mindspore/models/tree/r1.3/official/cv/densenet) |  ✅ |  ✅ | ✅   |    |   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DPN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/dpn) |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [EfficientNet-B0](https://gitee.com/mindspore/models/tree/r1.3/official/cv/efficientnet) |    |    |  ✅ |  ✅ |   |   |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [GoogLeNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/googlenet)    |  ✅  |  ✅ | ✅ |  ✅ |   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [InceptionV3](https://gitee.com/mindspore/models/tree/r1.3/official/cv/inceptionv3)   |  ✅ |    | ✅   |    |✅   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [InceptionV4](https://gitee.com/mindspore/models/tree/r1.3/official/cv/inceptionv4)    |  ✅ |    | ✅   |    | ✅  |   |    |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/lenet)              |  ✅ |  ✅ |  ✅ |  ✅ | ✅ | ✅ |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet (Quantization)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/lenet_quant)      |  ✅ |    |  ✅ |    |   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV1](https://gitee.com/mindspore/models/tree/r1.3/official/cv/mobilenetv1)      |  ✅ |    |  ✅  |    | ✅  |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV2](https://gitee.com/mindspore/models/tree/r1.3/official/cv/mobilenetv2)      |  ✅ |  ✅ |   |   | ✅ |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV2 (Quantization)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/mobilenetv2_quant)   |  ✅ |    |  ✅ |    |   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV3](https://gitee.com/mindspore/models/tree/r1.3/official/cv/mobilenetv3)  |    |    |  ✅ |  ✅ | ✅  |   |   |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [NASNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/nasnet) |    |    |  ✅ |  ✅ |   |   |  |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-18](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet)          |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet)          |  ✅ |  ✅ |  ✅ |  ✅ | ✅ |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50 (Quantization)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet50_quant)          |  ✅ |    |    |    |   |   |    |
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-101](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet)       |  ✅ |  ✅ | ✅ |  ✅ |   |   |✅ |
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNeXt50](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnext)     |  ✅ |    | ✅ |  ✅ |   |   |✅ |
|计算机视觉（CV）  | 图像分类（Image Classification）  | [SE-ResNet50](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet)      |  ✅ | ✅ |   |   |   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ShuffleNetV1](https://gitee.com/mindspore/models/tree/r1.3/official/cv/shufflenetv1)  |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ShuffleNetV2](https://gitee.com/mindspore/models/tree/r1.3/official/cv/shufflenetv2) |    |    |  ✅ |  ✅ |   |   |   |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [SqueezeNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/squeezenet) |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [Tiny-DarkNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/tinydarknet)       |   |    |    |    | ✅  |   |    |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [VGG16](https://gitee.com/mindspore/models/tree/r1.3/official/cv/vgg16)                |  ✅ |  ✅ |  ✅ |  ✅ |   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-152](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet152)             |  ✅ |  ✅  |    |    |   |   |    |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-34](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet)             |  ✅ |  ✅  |    |    |   |   |    |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [SimCLR](https://gitee.com/mindspore/models/tree/r1.3/official/cv/simclr)             |  ✅ |  ✅  |    |    |   |   |    |
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNeXt101](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnext)     |  ✅ |    | ✅ |  ✅ |   |   |✅ |
| 计算机视觉（CV） | 人脸识别（Face Recognition）  | [RetinaFace-ResNet50](https://gitee.com/mindspore/models/tree/r1.3/official/cv/retinaface_resnet50)     |    |    |  ✅ |  ✅ |   |   |   |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [CenterFace](https://gitee.com/mindspore/models/tree/r1.3/official/cv/centerface)     |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [CTPN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/ctpn)     |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Faster R-CNN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/faster_rcnn)   |  ✅ |    |  ✅ |    |   |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Mask R-CNN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/maskrcnn)         |  ✅ |   |    |    |   |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Mask R-CNN (MobileNetV1)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/maskrcnn_mobilenetv1)    |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD](https://gitee.com/mindspore/models/tree/r1.3/official/cv/ssd)                   |  ✅ |    | ✅ | ✅ | ✅ |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-MobileNetV1-FPN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/ssd)         |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-Resnet50-FPN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/ssd)                   |  ✅ |    |   |   |   |   |   |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-VGG16](https://gitee.com/mindspore/models/tree/r1.3/official/cv/ssd)                   |  ✅ |    |   |   |   |   |✅ |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [WarpCTC](https://gitee.com/mindspore/models/tree/r1.3/official/cv/warpctc)                    |  ✅ |    |  ✅ |    |  ✅  |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-ResNet18](https://gitee.com/mindspore/models/tree/r1.3/official/cv/yolov3_resnet18)    | ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-DarkNet53](https://gitee.com/mindspore/models/tree/r1.3/official/cv/yolov3_darknet53)         |  ✅ |  ✅ |  ✅ |  ✅ |   |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-DarkNet53 (Quantization)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/yolov3_darknet53_quant)  |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv4](https://gitee.com/mindspore/models/tree/r1.3/official/cv/yolov4)         |  ✅ | ✅   |    |    |   |   |    |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [CSPDarkNet53](https://gitee.com/mindspore/models/tree/r1.3/official/cv/cspdarknet53)         |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [RetinaNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/retinanet)         |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Faster R-CNN-ResnetV1-50](https://gitee.com/mindspore/models/tree/r1.3/official/cv/faster_rcnn)         |  ✅ |  ✅  |     |     |   |     |     |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Faster R-CNN-ResnetV1-101](https://gitee.com/mindspore/models/tree/r1.3/official/cv/faster_rcnn)         |  ✅ |  ✅  |     |     |   |     |     |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Faster R-CNN-ResnetV1-152](https://gitee.com/mindspore/models/tree/r1.3/official/cv/faster_rcnn)         |  ✅ |  ✅  |     |     |   |     |     |
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv5s](https://gitee.com/mindspore/models/tree/r1.3/official/cv/yolov5)         |  ✅ |  ✅  |     |     |   |     |     |
| 计算机视觉（CV） | 文本检测（Text Detection）  | [DeepText](https://gitee.com/mindspore/models/tree/r1.3/official/cv/deeptext)   |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV） | 文本检测（Text Detection）  | [PSENet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/psenet)   |  ✅ |  ✅ |    |    |   |   |✅ |
| 计算机视觉（CV） | 文本识别（Text Recognition）  | [CNN+CTC](https://gitee.com/mindspore/models/tree/r1.3/official/cv/cnnctc)                |  ✅ |  ✅ |    |    |   |   |✅ |
| 计算机视觉（CV） | 文本识别（Text Recognition）  | [CRNN-Seq2Seq-OCR](https://gitee.com/mindspore/models/tree/r1.3/official/cv/crnn_seq2seq_ocr)                |  ✅ |  ✅ |    |    |   |   |✅ |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [DeepLabV3](https://gitee.com/mindspore/models/tree/r1.3/official/cv/deeplabv3)     |  ✅ |    |    |    | ✅ |   |✅ |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [U-Net2D (Medical)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/unet)                |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [U-Net3D (Medical)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/unet3d)                |  ✅ |    |    |    |   |   |   |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [U-Net++](https://gitee.com/mindspore/models/tree/r1.3/official/cv/unet)                |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [FCN8s](https://gitee.com/mindspore/models/tree/r1.3/official/cv/FCN8s)                |  ✅ |    | ✅  |    |   |   |✅ |
| 计算机视觉（CV） | 关键点检测（Keypoint Detection）  | [OpenPose](https://gitee.com/mindspore/models/tree/r1.3/official/cv/openpose)                |  ✅ |    |    |    |   |   |   |
| 计算机视觉（CV） | 关键点检测（Keypoint Detection）  | [SimplePoseNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/simple_pose)                |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV） | 光学字符识别（Optical Character Recognition）  | [CRNN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/crnn)                |  ✅ |    | ✅    |    |   |   |   |
|  计算机视觉（CV） | 其它（Others） | [STGCN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/stgcn) |  ✅ |    ✅ |    |    |   |   |    |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [BERT](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/bert)        |  ✅ |  ✅ |  ✅ |  ✅ |   |   |  |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [FastText](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/fasttext)        |  ✅ |    |  ✅   |   |   |   |    |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [GNMT v2](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/gnmt_v2)        |  ✅ |    |    |   |   |   |    |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [GRU](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/gru)                |  ✅ |    |    |   |   |   |✅ |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [MASS](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/mass)   |  Ongoing |   |  Ongoing |   |   |   |Ongoing |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [SentimentNet](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/lstm)      |  ✅ |    |  ✅ |  ✅ | ✅ | ✅ |✅ |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [Transformer](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/transformer)   |  ✅ |  ✅ |  ✅ |  ✅ |   |   |    |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [TinyBERT](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/tinybert)       |  ✅ |  ✅ |  ✅ |   |   |   |    |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [TextCNN](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/textcnn)                |  ✅ |    |    |   |✅    |   |✅ |
| 自然语言处理（NLP） (NLP) | 情感分析（Sentiment Analysis）  | [EmoTect](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/emotect)  |  ✅ | ✅   |  ✅  |   |   |   |    |
| 自然语言处理（NLP） (NLP) | 问答对话（Dialogue）  | [DGU](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/dgu)  |  ✅ | ✅   |    |   |   |   |    |
|  推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction）  | [DeepFM](https://gitee.com/mindspore/models/tree/r1.3/official/recommend/deepfm)                               |  ✅ |  ✅ |  ✅ | ✅ | ✅ |   |✅ |
|  推荐（Recommender） | 推荐系统、搜索、排序（Recommender System, Search, Ranking）  | [Wide&Deep](https://gitee.com/mindspore/models/tree/r1.3/official/recommend/wide_and_deep)             |  ✅ |  ✅ |  ✅ | ✅ |   |   |✅ |
|  推荐（Recommender） | 推荐系统（Recommender System）  | [NAML](https://gitee.com/mindspore/models/tree/r1.3/official/recommend/naml)             |  ✅ |    |    |   |   |   |    |
|  推荐（Recommender） | 推荐系统（Recommender System）  | [NCF](https://gitee.com/mindspore/models/tree/r1.3/official/recommend/ncf)             |  ✅ |    |   |   |   |   |✅ |
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GCN](https://gitee.com/mindspore/models/tree/r1.3/official/gnn/gcn)  |  ✅ |  ✅ |    |    |   |   |✅ |
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GAT](https://gitee.com/mindspore/models/tree/r1.3/official/gnn/gat)  |  ✅ |  ✅ |    |    |   |   |✅ |
| 图神经网络（GNN） | 推荐系统（Recommender System） | [BGCF](https://gitee.com/mindspore/models/tree/r1.3/official/gnn/bgcf) |  ✅ |    |    |    |   |   |✅ |
| 强化学习（Reinforcement Learning） |  | [DQN](https://gitee.com/mindspore/models/tree/r1.3/official/rl/dqn)   |    |    |  ✅ |    |   |   |    |

### 研究网络

|  领域 | 子领域  | 网络   | Ascend（Graph） | Ascend（PyNative） | GPU（Graph） | GPU（PyNative） | CPU（Graph） | CPU（PyNative） | Ascend 310 |
|:------   |:------| :-----------   |:------:   |:------:   |:------:  |:------:  |:-----:  |:-----: |:-----: |
| 计算机视觉（CV） | 图像分类（Image Classification）  | [FaceAttributes](https://gitee.com/mindspore/models/tree/r1.3/research/cv/FaceAttribute)     |  ✅ |  ✅ |    |    |   |   |✅ |
| 计算机视觉（CV） | 图像分类（Image Classification）  | [FaceQualityAssessment](https://gitee.com/mindspore/models/tree/r1.3/research/cv/FaceQualityAssessment)     |  ✅ |  ✅ |  ✅   |    | ✅   |   |✅ |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [VGG19](https://gitee.com/mindspore/models/tree/r1.3/research/cv/vgg19)             |  ✅ |  ✅  |    |    |   |   |    |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNetV2-50](https://gitee.com/mindspore/models/tree/r1.3/research/cv/resnetv2)             |  ✅ |  ✅  |  ✅  |    |   |   |    |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNetV2-101](https://gitee.com/mindspore/models/tree/r1.3/research/cv/resnetv2)             |  ✅ |  ✅  |  ✅  |    |   |   |    |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNetV2-152](https://gitee.com/mindspore/models/tree/r1.3/research/cv/resnetv2)             |  ✅ |  ✅  |  ✅  |    |  
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [SE-Net](https://gitee.com/mindspore/models/tree/r1.3/research/cv/SE-Net)             |  ✅ |  ✅  |    |    |   |   |    |
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [SqueezeNet v1.1](https://gitee.com/mindspore/models/tree/r1.3/research/cv/squeezenet1_1)             |  ✅ |  ✅  |    |    |   |   |    |
| 计算机视觉（CV） | 人脸识别（Face Recognition）  | [FaceRecognition](https://gitee.com/mindspore/models/tree/r1.3/research/cv/FaceRecognition)     |  ✅ |    |    |    |   |   |
| 计算机视觉（CV） | 人脸识别（Face Recognition）  | [FaceRecognitionForTracking](https://gitee.com/mindspore/models/tree/r1.3/research/cv/FaceRecognitionForTracking)     |  ✅ |    |  ✅   |    |✅    |   |✅ |
| 计算机视觉（CV） | 人脸识别（Face Recognition）  | [ArcFace](https://gitee.com/mindspore/models/tree/r1.3/research/cv/arcface)     |  ✅ |  ✅  |     |    |    |     |     |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [FaceDetection](https://gitee.com/mindspore/models/tree/r1.3/research/cv/FaceDetection)     |  ✅ |    |  ✅   |    |✅    |   |✅ |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [SSD-GhostNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/ssd_ghostnet)               |  ✅ |    |    |    |   |   |✅ |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [Retinanet-Resnet101](https://gitee.com/mindspore/models/tree/r1.3/research/cv/retinanet_resnet101)               |  ✅ |✅    |    |    |   |   |    |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [IBN-Net](https://gitee.com/mindspore/models/tree/r1.3/research/cv/ibnnet)               |  ✅ |✅    |    |    |   |   |    |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [SSD-ResNet50-FPN](https://gitee.com/mindspore/models/tree/r1.3/research/cv/ssd_resnet50)               |  ✅ |✅    |    |    |   |   |    |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [WideResNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/wideresnet)               |  ✅ |✅    |    |    |   |   |    |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [Inception-ResnetV2](https://gitee.com/mindspore/models/tree/r1.3/research/cv/inception_resnet_v2)               |  ✅ |✅    |    |    |   |   |    |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [Learning To See In The Dark](https://gitee.com/mindspore/models/tree/r1.3/research/cv/LearningToSeeInTheDark)               |  ✅ |✅    |    |    |   |   |    |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [MnasNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/mnasnet)               |  ✅ |✅    |    |    |   |   |    |
| 计算机视觉（CV） | 目标检测（Object Detection）  | [MobileNetV3 Large](https://gitee.com/mindspore/models/tree/r1.3/research/cv/mobilenetv3_large)               |  ✅ |✅    |    |    |   |   |    |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [DeepLabv3+](https://gitee.com/mindspore/models/tree/r1.3/research/cv/deeplabv3plus)                |  ✅ |  ✅  |   |    |   |   |    |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [HarDNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/hardnet)                |  ✅ |  ✅  | ✅  |    |   |   |    |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [GloRe-Res50](https://gitee.com/mindspore/models/tree/r1.3/research/cv/glore_res50)                |  ✅ |  ✅  |   |    |   |   |    |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [GloRe-Res200](https://gitee.com/mindspore/models/tree/r1.3/research/cv/glore_res200)                |  ✅ |  ✅  | ✅  |    |   |   |    |
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [ICNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/ICNet)                |  ✅ |  ✅  | ✅  |    |   |   |    |
| 计算机视觉（CV）  | 关键点检测（Key Point Detection）  | [CenterNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/centernet)               |  ✅ |    |   |    | ✅ |   |    |
| 计算机视觉（CV）  | 关键点检测（Key Point Detection）  | [CenterNet-Det](https://gitee.com/mindspore/models/tree/r1.3/research/cv/centernet_det)               |  ✅ | ✅ |   |    |  |   |    |
| 计算机视觉（CV）  | 关键点检测（Key Point Detection）  | [CenterNet-Resnet101](https://gitee.com/mindspore/models/tree/r1.3/research/cv/centernet_resnet101)               |  ✅ | ✅ |   |    |  |   |    |
| 计算机视觉（CV）  | 图像风格迁移（Image Style Transfer）  | [CycleGAN](https://gitee.com/mindspore/models/tree/r1.3/research/cv/CycleGAN)             |  ✅  |    |   ✅ |  | ✅ |   |    |
| 计算机视觉（CV）  | 图像生成（Image Generation）  | [StarGAN](https://gitee.com/mindspore/models/tree/r1.3/research/cv/StarGAN)             |  ✅  | ✅  |    |  |  |   |    |
| 计算机视觉（CV）  | 图像生成（Image Generation）  | [WGAN](https://gitee.com/mindspore/models/tree/r1.3/research/cv/wgan)             |  ✅  | ✅  |    |  |  |   |    |
| 计算机视觉（CV）  | 图像生成（Image Generation）  | [Pix2Pix](https://gitee.com/mindspore/models/tree/r1.3/research/cv/Pix2Pix)             |  ✅  | ✅  |    |  |  |   |    |
| 计算机视觉（CV）  | 其它（Others）  | [GENet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/GENet_Res50)             |  ✅  | ✅  |    |  |  |   |    |
| 计算机视觉（CV）  | 其它（Others）  | [AutoAugment](https://gitee.com/mindspore/models/tree/r1.3/research/cv/autoaugment)             |  ✅  | ✅  |    |  |  |   |    |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [DS-CNN](https://gitee.com/mindspore/models/tree/r1.3/research/nlp/dscnn)      |  ✅ |  ✅ |    |   |   |   |✅ |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [TextRCNN](https://gitee.com/mindspore/models/tree/r1.3/research/nlp/textrcnn)  |  ✅ |    |    |   |   |   |✅ |
| 自然语言处理（NLP） (NLP) | 自然语言理解（Natural Language Understanding）  | [TPRR](https://gitee.com/mindspore/models/tree/r1.3/research/nlp/tprr)  |  ✅ |    |    |   |   |   |    |
|  推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction） | [AutoDis](https://gitee.com/mindspore/models/tree/r1.3/research/recommend/autodis)          |  ✅ |    |    |    |   |   |    |
|  推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction） | [FAT-DeepFFM](https://gitee.com/mindspore/models/tree/r1.3/research/recommend/Fat-DeepFFM)          |  ✅ | ✅ |    |    |   |   |    |
| 语音（Audio） | 音频标注（Audio Tagging） | [FCN-4](https://gitee.com/mindspore/models/tree/r1.3/research/audio/fcn-4)   |  ✅ |    |    |    |   |   |    |✅ |
| 语音（Audio） | 语音识别（Speech Recognition） | [DeepSpeech2](https://gitee.com/mindspore/models/tree/r1.3/research/audio/deepspeech2)   |   |    | ✅   |    | ✅  |   |    |
| 语音（Audio） | 语音合成（Generation） | [WaveNet](https://gitee.com/mindspore/models/tree/r1.3/research/audio/wavenet)    |   |    | ✅   |    | ✅  |   |    |
| 高性能计算（HPC） | 分子动力学（Molecular Dynamics） | [DeepPotentialH2O](https://gitee.com/mindspore/models/tree/r1.3/research/hpc/molecular_dynamics)   |  ✅ |  ✅ |    |    |   |   |    |
| 高性能计算（HPC） | 海洋模型（Ocean Model） | [GOMO](https://gitee.com/mindspore/models/tree/r1.3/research/hpc/ocean_model)   |    |    |  ✅ |    |   |   |    |

> 你也可以使用 [MindWizard工具](https://gitee.com/mindspore/mindinsight/tree/r1.3/mindinsight/wizard/) 快速生成经典网络脚本。
