# MindSpore网络支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `中级` `高级`

<!-- TOC -->

- [MindSpore网络支持](#mindspore网络支持)
    - [Model Zoo](#model-zoo)

<!-- /TOC -->

<a href="https://gitee.com/mindspore/docs/blob/master/docs/note/source_zh_cn/network_list_ms.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Model Zoo

|  领域 | 子领域  | 网络   | Ascend(Graph) | Ascend(PyNative) | GPU(Graph) | GPU(PyNaitve) | CPU(Graph) | CPU(PyNaitve)
|:----  |:-------  |:----   |:----    |:----    |:---- |:---- |:---- |:----
|计算机视觉（CV） | 图像分类（Image Classification）  | [AlexNet](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/alexnet/src/alexnet.py)   |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [GoogleNet](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/googlenet/src/googlenet.py)   |  Supported     |  Supported | Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/lenet/src/lenet.py)    |  Supported |  Supported |  Supported |  Supported | Supported | Supported
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet(量化)](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/lenet_quant/src/lenet_fusion.py)    |  Supported |  Doing |  Supported |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/resnet/src/resnet.py)   |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50(量化)](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/resnet50_quant/models/resnet_quant.py)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-101](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/resnet/src/resnet.py)        |  Supported |  Supported | Supported |  Supported | Doing | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [SE-ResNet50](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/resnet/src/resnet.py)       |  Supported | Supported | Doing | Doing | Doing | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNext50](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/resnext50/src/image_classification.py)    |  Supported |  Supported | Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [VGG16](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/vgg16/src/vgg.py)  |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [InceptionV3](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/inceptionv3/src/inception_v3.py) |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DenseNet121](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/densenet121/src/network/densenet.py) |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV2](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/mobilenetv2/src/mobilenetV2.py)        |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV2(量化)](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/mobilenetv2_quant/src/mobilenetV2.py)        |  Supported |  Doing |  Supported |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [MobileNetV3](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/mobilenetv3/src/mobilenetV3.py)        |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [NASNET](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/nasnet/src/nasnet_a_mobile.py) |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ShuffleNetV2](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/shufflenetv2/src/shufflenetv2.py) |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [EfficientNet-B0](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/efficientnet/src/efficientnet.py) |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [GhostNet](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/research/cv/ghostnet/src/ghostnet.py) |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet50-0.65x](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/research/cv/resnet50_adv_pruning/src/resnet_imgnet.py) |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [SSD-GhostNet](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/research/cv/ssd_ghostnet/src/ssd_ghostnet.py) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [TinyNet](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/research/cv/tinynet/src/tinynet.py) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
|计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/ssd/src/ssd.py)      |  Supported |  Supported |Supported |Supported | Supported | Supported
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YoloV3-ResNet18](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/yolov3_resnet18/src/yolov3.py)   |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YoloV3-DarkNet53](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/yolov3_darknet53/src/yolo.py)   |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YoloV3-DarkNet53(量化)](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/yolov3_darknet53_quant/src/darknet.py)   |  Supported |  Doing |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [FasterRCNN](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/faster_rcnn/src/FasterRcnn/faster_rcnn_r50.py)  |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [MaskRCNN](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/maskrcnn/src/maskrcnn/mask_rcnn_r50.py)  |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [WarpCTC](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/warpctc/src/warpctc.py)                    |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [Retinaface-ResNet50](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/retinaface_resnet50/src/network.py)   |  Doing |  Doing |  Supported |  Supported | Doing | Doing
| 计算机视觉 (CV) | 文本检测 (Text Detection)  | [PSENet](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/psenet/src/ETSNET/etsnet.py)                |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉 (CV) | 文本识别 (Text Recognition)  | [CNNCTC](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/cnnctc/src/cnn_ctc.py)                |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [DeeplabV3](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/deeplabv3/src/nets/deeplab_v3/deeplab_v3.py)   |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [UNet2D-Medical](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/cv/unet/src/unet/unet_model.py)   |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [BERT](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/nlp/bert/src/bert_model.py)  |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [Transformer](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/nlp/transformer/src/transformer_model.py)  |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [SentimentNet](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/nlp/lstm/src/lstm.py)    |  Doing |  Doing |  Supported |  Supported | Supported | Supported
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [MASS](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/nlp/mass/src/transformer/transformer_for_train.py)    |  Supported |  Supported |  Supported |  Supported | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TinyBert](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/nlp/tinybert/src/tinybert_model.py)   |  Supported |  Supported |  Supported | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [GNMT v2](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/nlp/gnmt_v2/src/gnmt_model/gnmt.py)    |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [DS-CNN](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/research/nlp/dscnn/src/ds_cnn.py)                |  Supported |  Doing |  Doing | Doing | Doing | Doing
| 推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction）  | [DeepFM](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/recommend/deepfm/src/deepfm.py)    |  Supported |  Supported |  Supported | Supported| Doing | Doing
| 推荐（Recommender） | 推荐系统、搜索、排序（Recommender System, Search, Ranking）  | [Wide&Deep](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/recommend/wide_and_deep/src/wide_and_deep.py)      |  Supported |  Supported |  Supported | Supported | Doing | Doing
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GCN](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/gnn/gcn/src/gcn.py)  |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GAT](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/gnn/gat/src/gat.py) |  Supported |  Supported |  Doing |  Doing | Doing | Doing
| 图神经网络（GNN） | 推荐系统（Recommender System） | [BGCF](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/gnn/bgcf/src/bgcf.py) |  Supported |  Doing |  Doing |  Doing | Doing | Doing
|语音（Audio） | 音频标注（Audio Tagging）  | [FCN-4](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/research/audio/fcn-4/src/musictagger.py)   |  Supported |  Supported |  Doing |  Doing | Doing | Doing

> 你也可以使用 [MindWizard工具](https://gitee.com/mindspore/mindinsight/tree/master/mindinsight/wizard/) 快速生成经典网络脚本。
