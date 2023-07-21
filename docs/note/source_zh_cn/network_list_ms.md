# MindSpore网络支持

`Linux` `Ascend` `GPU` `CPU` `模型开发` `中级` `高级`

[![查看源文件](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_zh_cn/network_list_ms.md)

## Model Zoo

|  领域 | 子领域  | 网络   | Ascend(Graph) | Ascend(PyNative) | GPU(Graph) | GPU(PyNaitve) | CPU(Graph)
|:----  |:-------  |:----   |:----    |:----    |:---- |:---- |:----
|计算机视觉（CV） | 图像分类（Image Classification）  | [AlexNet](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/alexnet/src/alexnet.py)   |  Supported |  Supported |  Supported |  Supported | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [GoogleNet](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/googlenet/src/googlenet.py)                             |  Supported     |  Supported | Supported |  Supported | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/lenet/src/lenet.py)    |  Supported |  Supported |  Supported |  Supported | Supported
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/resnet/src/resnet.py)   |  Supported |  Supported |  Supported |  Supported | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-101](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/resnet/src/resnet.py)                    |  Supported |  Supported | Supported |  Supported | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [SE-ResNet50](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/resnet/src/resnet.py)                    |  Supported | Doing | Doing | Doing | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNext50](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/resnext50/src/image_classification.py)         |  Supported |  Supported | Supported |  Supported | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [VGG16](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/vgg16/src/vgg.py)  |  Supported |  Supported |  Supported |  Supported | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [InceptionV3](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/inceptionv3/src/inception_v3.py) |  Supported |  Supported |  Doing |  Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [DenseNet121](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/densenet121/src/network/densenet.py) |  Supported |  Doing |  Doing |  Doing | Doing
| 计算机视觉（CV）  | 移动端图像分类（Mobile Image Classification）<br>目标检测（Object Detection）<br>语义分割（Semantic Segmentation）  | [MobileNetV2](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/mobilenetv2/src/mobilenetV2.py)        |  Supported |  Supported |  Supported |  Supported | Doing
| 计算机视觉（CV）  | 移动端图像分类（Mobile Image Classification）<br>目标检测（Object Detection）<br>语义分割（Semantic Segmentation）  | [MobileNetV3](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/mobilenetv3/src/mobilenetV3.py)        |  Doing |  Doing |  Supported |  Supported | Doing
|计算机视觉（CV）  | 目标检测（Object Detection）  | [SSD](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/ssd/src/ssd.py)      |  Supported |  Supported |Doing |Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YoloV3-ResNet18](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/yolov3_resnet18/src/yolov3.py)   |  Supported |  Doing |  Doing |  Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [YoloV3-DarkNet53](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/yolov3_darknet53/src/yolo.py)   |  Supported |  Doing |  Doing |  Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [FasterRCNN](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/faster_rcnn/src/FasterRcnn/faster_rcnn_r50.py)  |  Supported |  Doing |  Doing |  Doing | Doing
| 计算机视觉（CV）  | 目标检测（Object Detection）  | [MaskRCNN](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/maskrcnn/src/maskrcnn/mask_rcnn_r50.py)  |  Supported |  Doing |  Doing |  Doing | Doing
| 计算机视觉（CV） | 目标检测（Object Detection）  | [WarpCTC](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/warpctc/src/warpctc.py)                    |  Supported |  Doing |  Supported |  Supported | Doing
| 计算机视觉 (CV) | 文本检测 (Text Detection)  | [PSENet](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/psenet/src/ETSNET/etsnet.py)                |  Supported |  Doing |  Doing |  Doing | Doing
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [DeeplabV3](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/deeplabv3/src/nets/deeplab_v3/deeplab_v3.py)                    |  Supported |  Supported |  Doing |  Doing | Doing
| 计算机视觉（CV）  | 语义分割（Semantic Segmentation）  | [UNet2D-Medical](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/unet/src/unet/unet_model.py) |  Supported |  Doing |  Doing |  Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [BERT](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/nlp/bert/src/bert_model.py)  |  Supported |  Supported |  Supported |  Supported | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [Transformer](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/nlp/transformer/src/transformer_model.py)  |  Supported |  Doing |  Doing |  Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [SentimentNet](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/nlp/lstm/src/lstm.py)                          |  Doing |  Doing |  Supported |  Supported | Supported
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [MASS](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/nlp/mass/src/transformer/transformer_for_train.py)                     |  Supported |  Supported |  Doing |  Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TinyBert](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/nlp/tinybert/src/tinybert_model.py)                     |  Supported |  Doing |  Supported | Doing | Doing
| 推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction）  | [DeepFM](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/recommend/deepfm/src/deepfm.py)    |  Supported |  Supported |  Supported | Doing| Doing
| 推荐（Recommender） | 推荐系统、搜索、排序（Recommender System, Search ranking）  | [Wide&Deep](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/recommend/wide_and_deep/src/wide_and_deep.py)      |  Supported |  Supported |  Supported | Supported | Doing
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GCN](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/gnn/gcn/src/gcn.py)  |  Supported |  Doing |  Doing |  Doing | Doing
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GAT](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/gnn/gat/src/gat.py) |  Supported |  Doing |  Doing |  Doing | Doing

> 你也可以使用 [MindWizard工具](https://gitee.com/mindspore/mindinsight/tree/r1.0/mindinsight/wizard/) 快速生成经典网络脚本。
