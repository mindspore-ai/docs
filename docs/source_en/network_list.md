# Network List

`Ascend` `GPU` `CPU` `Model Development` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/tree/r0.7/docs/source_en/network_list.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Model Zoo

|  Domain | Sub Domain    | Network                                   | Ascend | GPU | CPU 
|:------   |:------| :-----------                               |:------   |:------  |:-----
|Computer Vision (CV) | Image Classification  | [AlexNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/alexnet/src/alexnet.py)          |  Supported |  Supported | Doing
| Computer Vision (CV)  | Image Classification  | [GoogleNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/googlenet/src/googlenet.py)                                               |  Doing     | Supported | Doing
| Computer Vision (CV)  | Image Classification  | [LeNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/lenet/src/lenet.py)              |  Supported |  Supported | Supported
| Computer Vision (CV)  | Image Classification  | [ResNet-50](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/resnet/src/resnet.py)          |  Supported |  Supported | Doing
|Computer Vision (CV)  | Image Classification  | [ResNet-101](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/resnet/src/resnet.py)                                              |  Supported |Doing | Doing
|Computer Vision (CV)  | Image Classification  | [ResNext50](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/resnext50/src/image_classification.py)                                             |  Supported | Supported | Doing
| Computer Vision (CV)  | Image Classification  | [VGG16](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/vgg16/src/vgg.py)                |  Supported |  Doing | Doing
| Computer Vision (CV)  | Image Classification  | [InceptionV3](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/inceptionv3/src/inception_v3.py)              |  Supported |  Doing | Doing
| Computer Vision (CV)  | Mobile Image Classification<br>Image Classification<br>Semantic Tegmentation  | [MobileNetV2](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/mobilenetv2/src/mobilenetV2.py)                   |  Supported |  Supported | Doing
| Computer Vision (CV)  | Mobile Image Classification<br>Image Classification<br>Semantic Tegmentation  | [MobileNetV3](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/mobilenetv3/src/mobilenetV3.py)                   |  Doing |  Supported | Doing
|Computer Vision (CV)  | Targets Detection  | [SSD](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/ssd/src/ssd.py)                   |  Supported |Doing | Doing
| Computer Vision (CV)  | Targets Detection  | [YoloV3-ResNet18](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/yolov3_resnet18/src/yolov3.py)         |  Supported |  Doing | Doing
| Computer Vision (CV)  | Targets Detection  | [YoloV3-DarkNet53](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/yolov3_darknet53/src/yolo.py)         |  Supported |  Doing | Doing
| Computer Vision (CV)  | Targets Detection  | [FasterRCNN](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/faster_rcnn/src/FasterRcnn/faster_rcnn_r50.py)         |  Supported |  Doing | Doing
| Computer Vision (CV) | Semantic Segmentation  | [DeeplabV3](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/deeplabv3/src/deeplabv3.py)                                           |  Supported |  Doing | Doing
| Computer Vision（CV） | Targets Detection  | [WarpCTC](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/warpctc/src/warpctc.py)                    |  Doing |  Supported | Doing
| Natural Language Processing (NLP) | Natural Language Understanding  | [BERT](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/bert/src/bert_model.py)                                          |  Supported |  Doing | Doing
| Natural Language Processing (NLP) | Natural Language Understanding  | [Transformer](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/transformer/src/transformer_model.py)                                          |  Supported |  Doing | Doing
| Natural Language Processing (NLP) | Natural Language Understanding  | [SentimentNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/lstm/src/lstm.py)                                          |  Doing |  Supported | Supported
| Natural Language Processing (NLP) | Natural Language Understanding  | [MASS](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/mass/src/transformer/transformer_for_train.py)                                          |  Supported |  Doing | Doing
| Natural Language Processing (NLP) | Natural Language Understanding  | [TinyBert](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/tinybert/src/tinybert_model.py)                                          |  Supported |  Doing | Doing
| Recommender | Recommender System, CTR prediction  | [DeepFM](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/recommend/deepfm/src/deepfm.py)                                          |  Supported |  Supported | Doing
| Recommender | Recommender System, Search ranking  | [Wide&Deep](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/recommend/wide_and_deep/src/wide_and_deep.py)                                          |  Supported |  Supported | Doing
| Graph Neural Networks（GNN）| Text Classification  | [GCN](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/gnn/gcn/src/gcn.py)                                          |  Supported |  Doing | Doing
| Graph Neural Networks（GNN）| Text Classification  | [GAT](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/gnn/gat/src/gat.py)                                          |  Supported |  Doing | Doing

> You can also use [MindWizard Tool](https://gitee.com/mindspore/mindinsight/tree/r0.7/mindinsight/wizard/) to quickly generate classic network scripts.

## Pre-trained Models
*It refers to the released MindSpore version. The hardware platforms that support model training are CPU, GPU and Ascend. As shown in the table below, ✓ indicates that the pre-trained model run on the selected platform.

Domain | Sub Domain| Network | Dataset | CPU   | GPU | Ascend | 0.5.0-beta* 
|:------   |:------ | :------- |:------ |:------   |:------  |:----- |:-----
|Computer Vision (CV) | Image Classification| [AlexNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/alexnet/src/alexnet.py) | CIFAR-10|    |    | ✓   |  [Download](http://download.mindspore.cn/model_zoo/official/cv/alexnet/alexnet_ascend_0.5.0_cifar10_official_classification_20200716.tar.gz)
|Computer Vision (CV) | Image Classification| [LeNet](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/lenet/src/lenet.py)| MNIST |   |   | ✓  | [Download](http://download.mindspore.cn/model_zoo/official/cv/lenet/lenet_ascend_0.5.0_mnist_official_classification_20200716.tar.gz)
|Computer Vision (CV) | Image Classification| [VGG16](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/vgg16/src/vgg.py)|    CIFAR-10 | |   | ✓ | [Download](http://download.mindspore.cn/model_zoo/official/cv/vgg/vgg16_ascend_0.5.0_cifar10_official_classification_20200715.tar.gz)
|Computer Vision (CV) | Image Classification| [ResNet-50](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/cv/resnet/src/resnet.py) | CIFAR-10|   |    | ✓ |[Download](http://download.mindspore.cn/model_zoo/official/cv/resnet/resnet50_v1.5_ascend_0.3.0_cifar10_official_classification_20200718.tar.gz)
|Computer Vision (CV)  | Targets Detection| [YoloV3-DarkNet53](https://gitee.com/mindspore/mindspore/tree/r0.7/model_zoo/official/cv/yolov3_darknet53/src/yolo.py) | COCO 2014|   |    | ✓  | [Download](http://download.mindspore.cn/model_zoo/official/cv/yolo/yolov3_darknet53_ascend_0.5.0_coco2014_official_object_detection_20200717.tar.gz) 
| Natural Language Processing (NLP) | Natural Language Understanding| [BERT_Base](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/bert/src/bert_model.py) | zhwiki |   |    | ✓  |  [Download](http://download.mindspore.cn/model_zoo/official/nlp/bert/bert_base_ascend_0.5.0_cn-wiki_official_nlp_20200720.tar.gz)
| Natural Language Processing (NLP) | Natural Language Understanding| [BERT_NEZHA](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/bert/src/bert_model.py)| zhwiki|  |    | ✓  |  [Download](http://download.mindspore.cn/model_zoo/official/nlp/bert/bert_nezha_ascend_0.5.0_cn-wiki_official_nlp_20200720.tar.gz) 
| Natural Language Processing (NLP) | Natural Language Understanding| [Transformer](https://gitee.com/mindspore/mindspore/blob/r0.7/model_zoo/official/nlp/transformer/src/transformer_model.py)| WMT English-German|   |   | ✓  | [Download](http://download.mindspore.cn/model_zoo/official/nlp/transformer/transformer_ascend_0.5.0_wmtende_official_machine_translation_20200713.tar.gz)
