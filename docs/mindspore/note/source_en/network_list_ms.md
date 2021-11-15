# MindSpore Network List

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Intermediate` `Expert`

<!-- TOC -->

- [MindSpore Network List](#mindspore-network-list)
    - [Model Zoo](#model-zoo)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/r1.3/docs/mindspore/note/source_en/network_list_ms.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.3/resource/_static/logo_source.png"></a>

## Model Zoo

### Official

|  Domain | Sub Domain    | Network  | Ascend (Graph) | Ascend (PyNative) | GPU (Graph) | GPU (PyNative) | CPU (Graph) | CPU (PyNative)|Ascend 310|
|:------   |:------| :-----------  |:------:   |:------:   |:------:  |:------:  |:-----: |:-----:|:-----:|
|Computer Vision (CV) | Image Classification  | [AlexNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/alexnet)          |   |   |  ✅ |  ✅ |   |   |   |
| Computer Vision (CV)  | Image Classification  | [CNN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/cnn_direction_model)  |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [DenseNet100](https://gitee.com/mindspore/models/tree/r1.3/official/cv/densenet) |    |    |    |    | ✅ | ✅ |   |
| Computer Vision (CV)  | Image Classification  | [DenseNet121](https://gitee.com/mindspore/models/tree/r1.3/official/cv/densenet) |  ✅ |  ✅ | ✅   |    |   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [DPN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/dpn) |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [EfficientNet-B0](https://gitee.com/mindspore/models/tree/r1.3/official/cv/efficientnet) |    |    |  ✅ |  ✅ |   |   |  |
| Computer Vision (CV)  | Image Classification  | [GoogLeNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/googlenet)    |  ✅  |  ✅ | ✅ |  ✅ |   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [InceptionV3](https://gitee.com/mindspore/models/tree/r1.3/official/cv/inceptionv3)   |  ✅ |    | ✅   |    |✅   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [InceptionV4](https://gitee.com/mindspore/models/tree/r1.3/official/cv/inceptionv4)    |  ✅ |    | ✅   |    | ✅  |   |    |
| Computer Vision (CV)  | Image Classification  | [LeNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/lenet)              |  ✅ |  ✅ |  ✅ |  ✅ | ✅ | ✅ |✅ |
| Computer Vision (CV)  | Image Classification  | [LeNet (Quantization)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/lenet_quant)      |  ✅ |    |  ✅ |    |   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [MobileNetV1](https://gitee.com/mindspore/models/tree/r1.3/official/cv/mobilenetv1)      |  ✅ |    |  ✅  |    | ✅  |   |✅ |
| Computer Vision (CV)  | Image Classification  | [MobileNetV2](https://gitee.com/mindspore/models/tree/r1.3/official/cv/mobilenetv2)      |  ✅ |  ✅ |   |   | ✅ |   |✅ |
| Computer Vision (CV)  | Image Classification  | [MobileNetV2 (Quantization)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/mobilenetv2_quant)   |  ✅ |    |  ✅ |    |   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [MobileNetV3](https://gitee.com/mindspore/models/tree/r1.3/official/cv/mobilenetv3)  |    |    |  ✅ |  ✅ | ✅  |   |   |
| Computer Vision (CV)  | Image Classification  | [NASNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/nasnet) |    |    |  ✅ |  ✅ |   |   |  |
| Computer Vision (CV)  | Image Classification  | [ResNet-18](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet)          |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [ResNet-50](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet)          |  ✅ |  ✅ |  ✅ |  ✅ | ✅ |   |✅ |
| Computer Vision (CV)  | Image Classification  | [ResNet-50 (Quantization)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet50_quant)          |  ✅ |    |    |    |   |   |    |
|Computer Vision (CV)  | Image Classification  | [ResNet-101](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet)       |  ✅ |  ✅ | ✅ |  ✅ |   |   |✅ |
|Computer Vision (CV)  | Image Classification  | [ResNeXt50](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnext)     |  ✅ |    | ✅ |  ✅ |   |   |✅ |
|Computer Vision (CV)  | Image Classification  | [SE-ResNet50](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet)      |  ✅ | ✅ |   |   |   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [ShuffleNetV1](https://gitee.com/mindspore/models/tree/r1.3/official/cv/shufflenetv1)  |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [ShuffleNetV2](https://gitee.com/mindspore/models/tree/r1.3/official/cv/shufflenetv2) |    |    |  ✅ |  ✅ |   |   |   |
| Computer Vision (CV)  | Image Classification  | [SqueezeNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/squeezenet) |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [Tiny-DarkNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/tinydarknet)       |   |    |    |    | ✅  |   |    |
| Computer Vision (CV)  | Image Classification  | [VGG16](https://gitee.com/mindspore/models/tree/r1.3/official/cv/vgg16)                |  ✅ |  ✅ |  ✅ |  ✅ |   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [ResNet-152](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet152)             |  ✅ |  ✅  |    |    |   |   |    |
| Computer Vision (CV)  | Image Classification  | [ResNet-34](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnet)             |  ✅ |  ✅  |    |    |   |   |    |
| Computer Vision (CV)  | Image Classification  | [SimCLR](https://gitee.com/mindspore/models/tree/r1.3/official/cv/simclr)             |  ✅ |  ✅  |    |    |   |   |    |
|Computer Vision (CV)  | Image Classification  | [ResNeXt101](https://gitee.com/mindspore/models/tree/r1.3/official/cv/resnext)     |  ✅ |    | ✅ |  ✅ |   |   |✅ |
| Computer Vision (CV) | Face Recognition  | [RetinaFace-ResNet50](https://gitee.com/mindspore/models/tree/r1.3/official/cv/retinaface_resnet50)   |     |    | ✅ | ✅ |   |   |   |
| Computer Vision (CV) | Object Detection  | [CenterFace](https://gitee.com/mindspore/models/tree/r1.3/official/cv/centerface)     |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV) | Object Detection  | [CTPN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/ctpn)     |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Object Detection  | [Faster R-CNN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/faster_rcnn)   |  ✅ |    |  ✅ |    |   |   |✅ |
| Computer Vision (CV)  | Object Detection  | [Mask R-CNN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/maskrcnn)         |  ✅ |   |    |    |   |   |✅ |
| Computer Vision (CV)  | Object Detection  | [Mask R-CNN (MobileNetV1)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/maskrcnn_mobilenetv1)    |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Object Detection  | [SSD](https://gitee.com/mindspore/models/tree/r1.3/official/cv/ssd)                   |  ✅ |    | ✅ | ✅ | ✅ |   |✅ |
| Computer Vision (CV)  | Object Detection  | [SSD-MobileNetV1-FPN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/ssd)         |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Object Detection  | [SSD-Resnet50-FPN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/ssd)                   |  ✅ |    |   |   |   |   |   |
| Computer Vision (CV)  | Object Detection  | [SSD-VGG16](https://gitee.com/mindspore/models/tree/r1.3/official/cv/ssd)                   |  ✅ |    |   |   |   |   |✅ |
| Computer Vision (CV) | Object Detection  | [WarpCTC](https://gitee.com/mindspore/models/tree/r1.3/official/cv/warpctc)                    |  ✅ |    |  ✅ |    |  ✅  |   |✅ |
| Computer Vision (CV)  | Object Detection  | [YOLOv3-ResNet18](https://gitee.com/mindspore/models/tree/r1.3/official/cv/yolov3_resnet18)    | ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Object Detection  | [YOLOv3-DarkNet53](https://gitee.com/mindspore/models/tree/r1.3/official/cv/yolov3_darknet53)         |  ✅ |  ✅ |  ✅ |  ✅ |   |   |✅ |
| Computer Vision (CV)  | Object Detection  | [YOLOv3-DarkNet53 (Quantization)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/yolov3_darknet53_quant)  |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Object Detection  | [YOLOv4](https://gitee.com/mindspore/models/tree/r1.3/official/cv/yolov4)         |  ✅ | ✅   |    |    |   |   |    |
| Computer Vision (CV)  | Object Detection  | [CSPDarkNet53](https://gitee.com/mindspore/models/tree/r1.3/official/cv/cspdarknet53)         |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Object Detection  | [RetinaNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/retinanet)         |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV)  | Object Detection  | [Faster R-CNN-ResnetV1-50](https://gitee.com/mindspore/models/tree/r1.3/official/cv/faster_rcnn)         |  ✅ |  ✅  |     |     |   |     |     |
| Computer Vision (CV)  | Object Detection  | [Faster R-CNN-ResnetV1-101](https://gitee.com/mindspore/models/tree/r1.3/official/cv/faster_rcnn)         |  ✅ |  ✅  |     |     |   |     |     |
| Computer Vision (CV)  | Object Detection  | [Faster R-CNN-ResnetV1-152](https://gitee.com/mindspore/models/tree/r1.3/official/cv/faster_rcnn)         |  ✅ |  ✅  |     |     |   |     |     |
| Computer Vision (CV)  | Object Detection  | [YOLOv5s](https://gitee.com/mindspore/models/tree/r1.3/official/cv/yolov5)         |  ✅ |  ✅  |     |     |   |     |     |
| Computer Vision (CV) | Text Detection  | [DeepText](https://gitee.com/mindspore/models/tree/r1.3/official/cv/deeptext)   |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV) | Text Detection  | [PSENet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/psenet)   |  ✅ |  ✅ |    |    |   |   |✅ |
| Computer Vision (CV) | Text Recognition  | [CNN+CTC](https://gitee.com/mindspore/models/tree/r1.3/official/cv/cnnctc)                |  ✅ |  ✅ |    |    |   |   |✅ |
| Computer Vision (CV) | Text Recognition  | [CRNN-Seq2Seq-OCR](https://gitee.com/mindspore/models/tree/r1.3/official/cv/crnn_seq2seq_ocr)                |  ✅ |  ✅ |    |    |   |   |✅ |
| Computer Vision (CV) | Semantic Segmentation  | [DeepLabV3](https://gitee.com/mindspore/models/tree/r1.3/official/cv/deeplabv3)     |  ✅ |    |    |    | ✅ |   |✅ |
| Computer Vision (CV) | Semantic Segmentation  | [U-Net2D (Medical)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/unet)                |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV) | Semantic Segmentation  | [U-Net3D (Medical)](https://gitee.com/mindspore/models/tree/r1.3/official/cv/unet3d)                |  ✅ |    |    |    |   |   |   |
| Computer Vision (CV) | Semantic Segmentation  | [U-Net++](https://gitee.com/mindspore/models/tree/r1.3/official/cv/unet)                |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV) | Semantic Segmentation  | [FCN8s](https://gitee.com/mindspore/models/tree/r1.3/official/cv/FCN8s)                |  ✅ |    | ✅  |    |   |   |✅ |
| Computer Vision (CV) | Keypoint Detection  | [OpenPose](https://gitee.com/mindspore/models/tree/r1.3/official/cv/openpose)                |  ✅ |    |    |    |   |   |   |
| Computer Vision (CV) | Keypoint Detection  | [SimplePoseNet](https://gitee.com/mindspore/models/tree/r1.3/official/cv/simple_pose)                |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV) | Optical Character Recognition  | [CRNN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/crnn)                |  ✅ |    | ✅    |    |   |   |   |
|  Computer Vision (CV) | Others | [STGCN](https://gitee.com/mindspore/models/tree/r1.3/official/cv/stgcn) |  ✅ |    ✅ |    |    |   |   |    |
| Natural Language Processing (NLP) | Natural Language Understanding  | [BERT](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/bert)        |  ✅ |  ✅ |  ✅ |  ✅ |   |   |  |
| Natural Language Processing (NLP) | Natural Language Understanding  | [FastText](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/fasttext)        |  ✅ |    |  ✅   |   |   |   |    |
| Natural Language Processing (NLP) | Natural Language Understanding  | [GNMT v2](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/gnmt_v2)        |  ✅ |    |    |   |   |   |    |
| Natural Language Processing (NLP) | Natural Language Understanding  | [GRU](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/gru)                |  ✅ |    |    |   |   |   |✅ |
| Natural Language Processing (NLP) | Natural Language Understanding  | [MASS](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/mass)   |  Ongoing |   |  Ongoing |   |   |   |Ongoing |
| Natural Language Processing (NLP) | Natural Language Understanding  | [SentimentNet](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/lstm)      |  ✅ |    |  ✅ |  ✅ | ✅ | ✅ |✅ |
| Natural Language Processing (NLP) | Natural Language Understanding  | [Transformer](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/transformer)   |  ✅ |  ✅ |  ✅ |  ✅ |   |   |    |
| Natural Language Processing (NLP) | Natural Language Understanding  | [TinyBERT](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/tinybert)       |  ✅ |  ✅ |  ✅ |   |   |   |    |
| Natural Language Processing (NLP) | Natural Language Understanding  | [TextCNN](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/textcnn)                |  ✅ |    |    |   |✅    |   |✅ |
| Natural Language Processing (NLP) | Sentiment Analysis  | [EmoTect](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/emotect)  |  ✅ | ✅   |  ✅  |   |   |   |    |
| Natural Language Processing (NLP) | Dialogue  | [DGU](https://gitee.com/mindspore/models/tree/r1.3/official/nlp/dgu)  |  ✅ | ✅   |    |   |   |   |    |
| Recommender | Recommender System, CTR prediction  | [DeepFM](https://gitee.com/mindspore/models/tree/r1.3/official/recommend/deepfm)                               |  ✅ |  ✅ |  ✅ | ✅ | ✅ |   |✅ |
| Recommender | Recommender System, Search, Ranking  | [Wide&Deep](https://gitee.com/mindspore/models/tree/r1.3/official/recommend/wide_and_deep)             |  ✅ |  ✅ |  ✅ | ✅ |   |   |✅ |
| Recommender | Recommender System  | [NAML](https://gitee.com/mindspore/models/tree/r1.3/official/recommend/naml)             |  ✅ |    |    |   |   |   |    |
| Recommender | Recommender System  | [NCF](https://gitee.com/mindspore/models/tree/r1.3/official/recommend/ncf)             |  ✅ |    |   |   |   |   |✅ |
| Graph Neural Networks (GNN) | Text Classification  | [GCN](https://gitee.com/mindspore/models/tree/r1.3/official/gnn/gcn)  |  ✅ |  ✅ |    |    |   |   |✅ |
| Graph Neural Networks (GNN) | Text Classification  | [GAT](https://gitee.com/mindspore/models/tree/r1.3/official/gnn/gat)  |  ✅ |  ✅ |    |    |   |   |✅ |
| Graph Neural Networks (GNN) | Recommender System | [BGCF](https://gitee.com/mindspore/models/tree/r1.3/official/gnn/bgcf) |  ✅ |    |    |    |   |   |✅ |
| Reinforcement Learning |  | [DQN](https://gitee.com/mindspore/models/tree/r1.3/official/rl/dqn)   |    |    |  ✅ |    |   |   |    |

### Research

|  Domain | Sub Domain    | Network  | Ascend (Graph) | Ascend (PyNative) | GPU (Graph) | GPU (PyNative)| CPU (Graph) | CPU (PyNative) |Ascend 310|
|:------   |:------| :-----------   |:------:   |:------:   |:------:  |:------:  |:-----:  |:-----: |:-----: |
| Computer Vision (CV) | Image Classification  | [FaceAttributes](https://gitee.com/mindspore/models/tree/r1.3/research/cv/FaceAttribute)     |  ✅ |  ✅ |    |    |   |   |✅ |
| Computer Vision (CV) | Image Classification  | [FaceQualityAssessment](https://gitee.com/mindspore/models/tree/r1.3/research/cv/FaceQualityAssessment)     |  ✅ |  ✅ |  ✅   |    | ✅   |   |✅ |
| Computer Vision (CV)  | Image Classification  | [VGG19](https://gitee.com/mindspore/models/tree/r1.3/research/cv/vgg19)             |  ✅ |  ✅  |    |    |   |   |    |
| Computer Vision (CV)  | Image Classification  | [ResNetV2-50](https://gitee.com/mindspore/models/tree/r1.3/research/cv/resnetv2)             |  ✅ |  ✅  |  ✅  |    |   |   |    |
| Computer Vision (CV)  | Image Classification  | [ResNetV2-101](https://gitee.com/mindspore/models/tree/r1.3/research/cv/resnetv2)             |  ✅ |  ✅  |  ✅  |    |   |   |    |
| Computer Vision (CV)  | Image Classification  | [ResNetV2-152](https://gitee.com/mindspore/models/tree/r1.3/research/cv/resnetv2)             |  ✅ |  ✅  |  ✅  |    |  
| Computer Vision (CV)  | Image Classification  | [SE-Net](https://gitee.com/mindspore/models/tree/r1.3/research/cv/SE-Net)             |  ✅ |  ✅  |    |    |   |   |    |
| Computer Vision (CV)  | Image Classification  | [SqueezeNet v1.1](https://gitee.com/mindspore/models/tree/r1.3/research/cv/squeezenet1_1)             |  ✅ |  ✅  |    |    |   |   |    |
| Computer Vision (CV) | Face Recognition  | [FaceRecognition](https://gitee.com/mindspore/models/tree/r1.3/research/cv/FaceRecognition)     |  ✅ |    |    |    |   |   |
| Computer Vision (CV) | Face Recognition  | [FaceRecognitionForTracking](https://gitee.com/mindspore/models/tree/r1.3/research/cv/FaceRecognitionForTracking)     |  ✅ |    |  ✅   |    |✅    |   |✅ |
| Computer Vision (CV) | Face Recognition  | [ArcFace](https://gitee.com/mindspore/models/tree/r1.3/research/cv/arcface)     |  ✅ |  ✅  |     |    |    |     |     |
| Computer Vision (CV) | Object Detection  | [FaceDetection](https://gitee.com/mindspore/models/tree/r1.3/research/cv/FaceDetection)     |  ✅ |    |  ✅   |    |✅    |   |✅ |
| Computer Vision (CV) | Object Detection  | [SSD-GhostNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/ssd_ghostnet)               |  ✅ |    |    |    |   |   |✅ |
| Computer Vision (CV) | Object Detection  | [Retinanet-Resnet101](https://gitee.com/mindspore/models/tree/r1.3/research/cv/retinanet_resnet101)               |  ✅ |✅    |    |    |   |   |    |
| Computer Vision (CV) | Object Detection  | [IBN-Net](https://gitee.com/mindspore/models/tree/r1.3/research/cv/ibnnet)               |  ✅ |✅    |    |    |   |   |    |
| Computer Vision (CV) | Object Detection  | [SSD-ResNet50-FPN](https://gitee.com/mindspore/models/tree/r1.3/research/cv/ssd_resnet50)               |  ✅ |✅    |    |    |   |   |    |
| Computer Vision (CV) | Object Detection  | [WideResNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/wideresnet)               |  ✅ |✅    |    |    |   |   |    |
| Computer Vision (CV) | Object Detection  | [Inception-ResnetV2](https://gitee.com/mindspore/models/tree/r1.3/research/cv/inception_resnet_v2)               |  ✅ |✅    |    |    |   |   |    |
| Computer Vision (CV) | Object Detection  | [Learning To See In The Dark](https://gitee.com/mindspore/models/tree/r1.3/research/cv/LearningToSeeInTheDark)               |  ✅ |✅    |    |    |   |   |    |
| Computer Vision (CV) | Object Detection  | [MnasNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/mnasnet)               |  ✅ |✅    |    |    |   |   |    |
| Computer Vision (CV) | Object Detection  | [MobileNetV3 Large](https://gitee.com/mindspore/models/tree/r1.3/research/cv/mobilenetv3_large)               |  ✅ |✅    |    |    |   |   |    |
| Computer Vision (CV) | Semantic Segmentation  | [DeepLabv3+](https://gitee.com/mindspore/models/tree/r1.3/research/cv/deeplabv3plus)                |  ✅ |  ✅  |   |    |   |   |    |
| Computer Vision (CV) | Semantic Segmentation  | [HarDNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/hardnet)                |  ✅ |  ✅  | ✅  |    |   |   |    |
| Computer Vision (CV) | Semantic Segmentation  | [GloRe-Res50](https://gitee.com/mindspore/models/tree/r1.3/research/cv/glore_res50)                |  ✅ |  ✅  |   |    |   |   |    |
| Computer Vision (CV) | Semantic Segmentation  | [GloRe-Res200](https://gitee.com/mindspore/models/tree/r1.3/research/cv/glore_res200)                |  ✅ |  ✅  | ✅  |    |   |   |    |
| Computer Vision (CV) | Semantic Segmentation  | [ICNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/ICNet)                |  ✅ |  ✅  | ✅  |    |   |   |    |
| Computer Vision (CV)  | Key Point Detection  | [CenterNet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/centernet)               |  ✅ |    |   |    | ✅ |   |    |
| Computer Vision (CV)  | Key Point Detection  | [CenterNet-Det](https://gitee.com/mindspore/models/tree/r1.3/research/cv/centernet_det)               |  ✅ | ✅ |   |    |  |   |    |
| Computer Vision (CV)  | Key Point Detection  | [CenterNet-Resnet101](https://gitee.com/mindspore/models/tree/r1.3/research/cv/centernet_resnet101)               |  ✅ | ✅ |   |    |  |   |    |
| Computer Vision (CV)  | Image Style Transfer  | [CycleGAN](https://gitee.com/mindspore/models/tree/r1.3/research/cv/CycleGAN)             |  ✅  |    |   ✅ |  | ✅ |   |    |
| Computer Vision (CV)  | Image Generation  | [StarGAN](https://gitee.com/mindspore/models/tree/r1.3/research/cv/StarGAN)             |  ✅  | ✅  |    |  |  |   |    |
| Computer Vision (CV)  | Image Generation  | [WGAN](https://gitee.com/mindspore/models/tree/r1.3/research/cv/wgan)             |  ✅  | ✅  |    |  |  |   |    |
| Computer Vision (CV)  | Image Generation  | [Pix2Pix](https://gitee.com/mindspore/models/tree/r1.3/research/cv/Pix2Pix)             |  ✅  | ✅  |    |  |  |   |    |
| Computer Vision (CV)  | Others  | [GENet](https://gitee.com/mindspore/models/tree/r1.3/research/cv/GENet_Res50)             |  ✅  | ✅  |    |  |  |   |    |
| Computer Vision (CV)  | Others  | [AutoAugment](https://gitee.com/mindspore/models/tree/r1.3/research/cv/autoaugment)             |  ✅  | ✅  |    |  |  |   |    |
| Natural Language Processing (NLP) | Natural Language Understanding  | [DS-CNN](https://gitee.com/mindspore/models/tree/r1.3/research/nlp/dscnn)      |  ✅ |  ✅ |    |   |   |   |✅ |
| Natural Language Processing (NLP) | Natural Language Understanding  | [TextRCNN](https://gitee.com/mindspore/models/tree/r1.3/research/nlp/textrcnn)  |  ✅ |    |    |   |   |   |✅ |
| Natural Language Processing (NLP) | Natural Language Understanding  | [TPRR](https://gitee.com/mindspore/models/tree/r1.3/research/nlp/tprr)  |  ✅ |    |    |   |   |   |    |
| Recommender | Recommender System, CTR prediction | [AutoDis](https://gitee.com/mindspore/models/tree/r1.3/research/recommend/autodis)          |  ✅ |    |    |    |   |   |    |
| Recommender | Recommender System, CTR prediction | [FAT-DeepFFM](https://gitee.com/mindspore/models/tree/r1.3/research/recommend/Fat-DeepFFM)          |  ✅ | ✅ |    |    |   |   |    |
| Audio | Audio Tagging | [FCN-4](https://gitee.com/mindspore/models/tree/r1.3/research/audio/fcn-4)   |  ✅ |    |    |    |   |   |    |✅ |
| Audio |Speech Recognition | [DeepSpeech2](https://gitee.com/mindspore/models/tree/r1.3/research/audio/deepspeech2)   |   |    | ✅   |    | ✅  |   |    |
| Audio |Generation | [WaveNet](https://gitee.com/mindspore/models/tree/r1.3/research/audio/wavenet)    |   |    | ✅   |    | ✅  |   |    |
| High Performance Computing | Molecular Dynamics | [DeepPotentialH2O](https://gitee.com/mindspore/models/tree/r1.3/research/hpc/molecular_dynamics)   |  ✅ |  ✅ |    |    |   |   |    |
| High Performance Computing | Ocean Model | [GOMO](https://gitee.com/mindspore/models/tree/r1.3/research/hpc/ocean_model)   |    |    |  ✅ |    |   |   |    |

> You can also use [MindWizard Tool](https://gitee.com/mindspore/mindinsight/tree/r1.3/mindinsight/wizard/) to quickly generate classic network scripts.
