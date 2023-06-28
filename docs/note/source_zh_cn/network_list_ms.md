# MindSpore网络支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `中级` `高级`

<a href="https://gitee.com/mindspore/docs/blob/r1.1/docs/note/source_zh_cn/network_list_ms.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Model Zoo

### 标准网络

|  领域 | 子领域  | 网络   | Ascend（Graph） | Ascend（PyNative） | GPU（Graph） | GPU（PyNative） | CPU（Graph） | CPU（PyNative）
|:----  |:-------  |:----   |:----    |:----    |:---- |:---- |:---- |:----
|计算机视觉（CV） | 图像分类（Image Classification）  | [AlexNet](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/alexnet/src/alexnet.py)   |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [CNN](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/cnn_direction_model/src/cnn_direction_model.py)  |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DenseNet121](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/densenet121/src/network/densenet.py) |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [EfficientNet-B0](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/efficientnet/src/efficientnet.py) |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [GoogLeNet](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/googlenet/src/googlenet.py)   |  Supported     |  Supported | Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [InceptionV3](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/inceptionv3/src/inception_v3.py) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [InceptionV4](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/inceptionv4/src/inceptionv4.py) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/lenet/src/lenet.py)    |  Supported |  Supported |  Supported |  Supported | Supported | Supported
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet（量化）](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/lenet_quant/src/lenet_fusion.py)    |  Supported |  Doing |  Supported |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV1](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/mobilenetv1/src/mobilenet_v1.py)        |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV2](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/mobilenetv2/src/mobilenetV2.py)        |  Supported |  Supported |  Supported |  Supported | Supported | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV2（量化）](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/mobilenetv2_quant/src/mobilenetV2.py)        |  Supported |  Doing |  Supported |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV3](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/mobilenetv3/src/mobilenetV3.py)        |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [NASNet](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/nasnet/src/nasnet_a_mobile.py) |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/resnet/src/resnet.py)   |  Supported |  Supported |  Supported |  Supported | Supported | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50（量化）](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/resnet50_quant/models/resnet_quant.py)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-101](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/resnet/src/resnet.py)        |  Supported |  Supported | Supported |  Supported | Doing | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNeXt50](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/resnext50/src/image_classification.py)    |  Supported |  Doing | Supported |  Supported | Doing | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [SE-ResNet50](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/resnet/src/resnet.py)       |  Supported | Supported | Doing | Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ShuffleNetV1](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/shufflenetv1/src/shufflenetv1.py)        |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ShuffleNetV2](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/shufflenetv2/src/shufflenetv2.py) |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  |[SqueezeNet](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/squeezenet/src/squeezenet.py) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [Tiny-DarkNet](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/tinydarknet/src/tinydarknet.py)  |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [VGG16](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/vgg16/src/vgg.py)  |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [Xception](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/xception/src/Xception.py) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [CenterFace](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/centerface/src/centerface.py)     |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [CTPN](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/ctpn/src/ctpn.py)     |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Faster R-CNN](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/faster_rcnn/src/FasterRcnn/faster_rcnn_r50.py)  |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [Mask R-CNN](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/maskrcnn/src/maskrcnn/mask_rcnn_r50.py)  |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  |[Mask R-CNN (MobileNetV1)](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/maskrcnn_mobilenetv1/src/maskrcnn_mobilenetv1/mobilenetv1.py)         |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [RetinaFace-ResNet50](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/retinaface_resnet50/src/network.py)   |  Doing |  Doing |  Supported |  Supported | Doing | Doing
|计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/ssd/src/ssd.py)      |  Supported |  Doing |Supported |Supported | Supported | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  |[SSD-MobileNetV1-FPN](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/ssd/src/mobilenet_v1_fpn.py)         |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [WarpCTC](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/warpctc/src/warpctc.py)                    |  Supported |  Doing |  Supported |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-ResNet18](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/yolov3_resnet18/src/yolov3.py)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-DarkNet53](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/yolov3_darknet53/src/yolo.py)   |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YOLOv3-DarkNet53（量化）](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/yolov3_darknet53_quant/src/darknet.py)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  |[YOLOv4](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/yolov4/src/yolo.py)         |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 文本检测（Text Detection）  | [AdvancedEAST](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/research/cv/advanced_east)                |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 文本检测（Text Detection）  | [DeepText](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/deeptext/src/Deeptext/deeptext_vgg16.py)                |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 文本检测（Text Detection）  | [EAST](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/official/cv/east)                |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 文本检测（Text Detection）  | [PSENet](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/psenet/src/ETSNET/etsnet.py)                |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 文本识别（Text Recognition）  | [CNN+CTC](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/cnnctc/src/cnn_ctc.py)                |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [DeepLabV3](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/deeplabv3/src/nets/deeplab_v3/deeplab_v3.py)   |  Supported |  Doing |  Doing |  Doing | Supported | Doing
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [U-Net2D (Medical)](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/unet/src/unet/unet_model.py)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 关键点检测（Keypoint Detection）  |[OpenPose](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/openpose/src/openposenet.py)                |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 关键点检测（Keypoint Detection）  |[SimplePoseNet](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/simple_pose/src/model.py)                |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 光学字符识别（Optical Character Recognition）  |[CRNN](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/cv/crnn/src/crnn.py)                |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [BERT](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/nlp/bert/src/bert_model.py)  |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [FastText](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/nlp/fasttext/src/fasttext_model.py)    |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [GNMT v2](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/nlp/gnmt_v2/src/gnmt_model/gnmt.py)    |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [GRU](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/nlp/gru/src/seq2seq.py)            |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [MASS](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/nlp/mass/src/transformer/transformer_for_train.py)    |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [SentimentNet (LSTM)](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/nlp/lstm/src/lstm.py)    |  Supported |  Doing |  Supported |  Supported | Supported | Supported
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [Transformer](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/nlp/transformer/src/transformer_model.py)  |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TinyBERT](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/nlp/tinybert/src/tinybert_model.py)   |  Supported |  Supported |  Supported | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TextCNN](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/nlp/textcnn/src/textcnn.py)            |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction）  | [DeepFM](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/recommend/deepfm/src/deepfm.py)    |  Supported |  Supported |  Supported | Supported| Supported | Doing
| 推荐（Recommender） | 推荐系统、搜索、排序（Recommender System, Search, Ranking）  | [Wide&Deep](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/recommend/wide_and_deep/src/wide_and_deep.py)      |  Supported |  Supported |  Supported | Supported | Doing | Doing
| 推荐（Recommender） | 推荐系统（Recommender System）  | [NCF](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/recommend/ncf/src/ncf.py)    |  Supported |  Doing |  Supported | Doing| Doing | Doing
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GCN](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/gnn/gcn/src/gcn.py)  |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GAT](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/gnn/gat/src/gat.py) |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 图神经网络（GNN） | 推荐系统（Recommender System） | [BGCF](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/official/gnn/bgcf/src/bgcf.py) |  Supported |  Doing |  Doing |  Doing | Doing | Doing

### 研究网络

|  领域 | 子领域  | 网络   | Ascend（Graph） | Ascend（PyNative） | GPU（Graph） | GPU（PyNative） | CPU（Graph） | CPU（PyNative）
|:----  |:-------  |:----   |:----    |:----    |:---- |:---- |:---- |:----
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceAttributes](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/cv/FaceAttribute/src/FaceAttribute/resnet18.py)     |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [FaceDetection](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/cv/FaceDetection/src/FaceDetection/yolov3.py)  |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceQualityAssessment](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/cv/FaceQualityAssessment/src/face_qa.py)     |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceRecognition](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/cv/FaceRecognition/src/init_network.py)     |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 图像分类（Image Classification）  |[FaceRecognitionForTracking](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/cv/FaceRecognitionForTracking/src/reid.py)     |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD-GhostNet](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/cv/ssd_ghostnet/src/ssd_ghostnet.py)           |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 关键点检测（Key Point Detection）  | [CenterNet](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/cv/centernet/src/centernet_pose.py)          |  Supported |  Doing | Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像生成（Image Generation）  | [CycleGAN](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo/research/cv/CycleGAN)       |  Supported     |  Doing | Doing |  Supported | Supported | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [DS-CNN](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/nlp/dscnn/src/ds_cnn.py)          |  Supported |  Supported |  Doing | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TextRCNN](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/nlp/textrcnn/src/textrcnn.py)    |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction） | [AutoDis](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/recommend/autodis/src/autodis.py)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
|语音（Audio） | 音频标注（Audio Tagging）  | [FCN-4](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/audio/fcn-4/src/musictagger.py)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
|高性能计算（HPC） | 分子动力学（Molecular Dynamics）  |  [DeepPotentialH2O](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/hpc/molecular_dynamics/src/network.py)   |  Supported | Supported|  Doing |  Doing | Doing | Doing
|高性能计算（HPC） | 海洋模型（Ocean Model）  |  [GOMO](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/hpc/ocean_model/src/GOMO.py)   |  Doing |  Doing |  Supported |  Doing | Doing | Doing

> 你也可以使用 [MindWizard工具](https://gitee.com/mindspore/mindinsight/tree/r1.1/mindinsight/wizard/) 快速生成经典网络脚本。
