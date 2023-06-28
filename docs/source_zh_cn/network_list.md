# 网络支持

`Ascend` `GPU` `CPU` `模型开发` `中级` `高级`
 
<a href="https://gitee.com/mindspore/docs/tree/r0.7/docs/source_zh_cn/network_list.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Model Zoo

|  领域 | 子领域  | 网络   | Ascend | GPU | CPU 
|:----  |:-------  |:----   |:----    |:---- |:----
|计算机视觉（CV） | 图像分类（Image Classification）  | [AlexNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/alexnet/src/alexnet.py)   |  Supported |  Supported | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [GoogleNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/googlenet/src/googlenet.py)                             |  Doing     | Supported | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [LeNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/lenet/src/lenet.py)    |  Supported |  Supported | Supported
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-50](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/resnet/src/resnet.py)   |  Supported |  Supported | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNet-101](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/resnet/src/resnet.py)                    |  Supported |Doing | Doing
|计算机视觉（CV）  | 图像分类（Image Classification）  | [ResNext50](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/resnext50/src/image_classification.py)         |  Supported | Supported | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [VGG16](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/vgg16/src/vgg.py)  |  Supported |  Doing | Doing
| 计算机视觉（CV）  | 图像分类（Image Classification）  | [InceptionV3](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/inceptionv3/src/inception_v3.py) |  Supported |  Doing | Doing
| 计算机视觉（CV）  | 移动端图像分类（Mobile Image Classification）<br>目标检测（Image Classification）<br>语义分割（Semantic Tegmentation）  | [MobileNetV2](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/mobilenetv2/src/mobilenetV2.py)        |  Supported |  Supported | Doing
| 计算机视觉（CV）  | 移动端图像分类（Mobile Image Classification）<br>目标检测（Image Classification）<br>语义分割（Semantic Tegmentation）  | [MobileNetV3](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/mobilenetv3/src/mobilenetV3.py)        |  Doing |  Supported | Doing
|计算机视觉（CV）  | 目标检测（Targets Detection）  | [SSD](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/ssd/src/ssd.py)      |  Supported |Doing | Doing
| 计算机视觉（CV）  | 目标检测（Targets Detection）  | [YoloV3-ResNet18](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/yolov3_resnet18/src/yolov3.py)   |  Supported |  Doing | Doing
| 计算机视觉（CV）  | 目标检测（Targets Detection）  | [YoloV3-DarkNet53](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/yolov3_darknet53/src/yolo.py)   |  Supported |  Doing | Doing
| 计算机视觉（CV）  | 目标检测（Targets Detection）  | [FasterRCNN](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/faster_rcnn/src/FasterRcnn/faster_rcnn_r50.py)  |  Supported |  Doing | Doing
| 计算机视觉（CV） | 语义分割（Semantic Segmentation）  | [DeeplabV3](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/deeplabv3/src/deeplabv3.py)                    |  Supported |  Doing | Doing
| 计算机视觉（CV） | 目标检测（Targets Detection）  | [WarpCTC](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/warpctc/src/warpctc.py)                    |  Doing |  Supported | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [BERT](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/bert/src/bert_model.py)  |  Supported |  Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [Transformer](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/transformer/src/transformer_model.py)  |  Supported |  Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [SentimentNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/lstm/src/lstm.py)                          |  Doing |  Supported | Supported
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [MASS](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/mass/src/transformer/transformer_for_train.py)                     |  Supported |  Doing | Doing
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）  | [TinyBert](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/tinybert/src/tinybert_model.py)                     |  Supported |  Doing | Doing
| 推荐（Recommender） | 推荐系统、点击率预估（Recommender System, CTR prediction）  | [DeepFM](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/recommend/deepfm/src/deepfm.py)    |  Supported |  Supported | Doing
| 推荐（Recommender） | 推荐系统、搜索、排序（Recommender System, Search ranking）  | [Wide&Deep](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/recommend/wide_and_deep/src/wide_and_deep.py)      |  Supported |  Supported | Doing
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GCN](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/gnn/gcn/src/gcn.py)  |  Supported |  Doing | Doing
| 图神经网络（GNN） | 文本分类（Text Classification）  | [GAT](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/gnn/gat/src/gat.py) |  Supported |  Doing | Doing

> 你也可以使用 [MindWizard工具](https://gitee.com/mindspore/mindinsight/tree/r0.7/mindinsight/wizard/) 快速生成经典网络脚本。

## 预训练模型
*代表MindSpore已发布的版本号，支持网络训练的硬件平台有CPU、GPU和Ascend，以下表格中 ✓ 代表模型是基于选中的硬件平台训练而来。

|  领域 | 子领域  | 网络 |数据集 | CPU   | GPU | Ascend | 0.5.0-beta* 
|:----  |:-----  |:----   |:---- |:----    |:---- |:---- |:------
|计算机视觉（CV） | 图像分类（Image Classification） | [AlexNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/alexnet/src/alexnet.py) | CIFAR-10  |  |    | ✓  |  [下载](http://download.mindspore.cn/model_zoo/official/cv/alexnet/alexnet_ascend_0.5.0_cifar10_official_classification_20200716.tar.gz)
|计算机视觉（CV） | 图像分类（Image Classification）| [LeNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/lenet/src/lenet.py)| MNIST |  |   | ✓  |   [下载](http://download.mindspore.cn/model_zoo/official/cv/lenet/lenet_ascend_0.5.0_mnist_official_classification_20200716.tar.gz)
|计算机视觉（CV） | 图像分类（Image Classification）| [VGG16](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/vgg16/src/vgg.py)|CIFAR-10  |  |   | ✓  | [下载](http://download.mindspore.cn/model_zoo/official/cv/vgg/vgg16_ascend_0.5.0_cifar10_official_classification_20200715.tar.gz)
|计算机视觉（CV） | 图像分类（Image Classification）| [ResNet-50](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/resnet/src/resnet.py) |CIFAR-10 |   |    | ✓  |[下载](http://download.mindspore.cn/model_zoo/official/cv/resnet/resnet50_v1.5_ascend_0.3.0_cifar10_official_classification_20200718.tar.gz)
|计算机视觉（CV）  | 目标检测（Targets Detection）| [YoloV3-DarkNet53](https://gitee.com/mindspore/mindspore/tree/r0.7/model_zoo/official/cv/yolov3_darknet53) |COCO 2014  | |    | ✓  | [下载](http://download.mindspore.cn/model_zoo/official/cv/yolo/yolov3_darknet53_ascend_0.5.0_coco2014_official_object_detection_20200717.tar.gz) 
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）| [BERT_Base](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/bert/src/bert_model.py) | zhwiki |   |    | ✓  |  [下载](http://download.mindspore.cn/model_zoo/official/nlp/bert/bert_base_ascend_0.5.0_cn-wiki_official_nlp_20200720.tar.gz)
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）| [BERT_NEZHA](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/bert/src/bert_model.py)| zhwiki |  |    | ✓  | [下载](http://download.mindspore.cn/model_zoo/official/nlp/bert/bert_nezha_ascend_0.5.0_cn-wiki_official_nlp_20200720.tar.gz) 
| 自然语言处理（NLP） | 自然语言理解（Natural Language Understanding）| [Transformer](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/transformer/src/transformer_model.py)|WMT English-German  |   |   | ✓  | [下载](http://download.mindspore.cn/model_zoo/official/nlp/transformer/transformer_ascend_0.5.0_wmtende_official_machine_translation_20200713.tar.gz)
