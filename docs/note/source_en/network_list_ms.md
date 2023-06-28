# MindSpore Network List

`Linux` `Ascend` `GPU` `CPU` `Model Development` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/r1.0/docs/note/source_en/network_list_ms.md" target="_blank"><img src="./_static/logo_source.png"></a>

## Model Zoo

|  Domain | Sub Domain    | Network                                   | Ascend(Graph) | Ascend(PyNative) | GPU(Graph) | GPU(PyNative)| CPU(Graph) 
|:------   |:------| :-----------                               |:------   |:------   |:------  |:------  |:-----
|Computer Vision (CV) | Image Classification  | [AlexNet](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/alexnet/src/alexnet.py)          |  Supported |  Supported |  Supported |  Supported | Doing
| Computer Vision (CV)  | Image Classification  | [GoogleNet](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/googlenet/src/googlenet.py)                                               |  Supported     |  Supported | Supported |  Supported | Doing
| Computer Vision (CV)  | Image Classification  | [LeNet](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/lenet/src/lenet.py)              |  Supported |  Supported |  Supported |  Supported | Supported
| Computer Vision (CV)  | Image Classification  | [ResNet-50](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/resnet/src/resnet.py)          |  Supported |  Supported |  Supported |  Supported | Doing
|Computer Vision (CV)  | Image Classification  | [ResNet-101](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/resnet/src/resnet.py)                                              |  Supported |  Supported | Supported |  Supported | Doing
|Computer Vision (CV)  | Image Classification  | [SE-ResNet50](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/resnet/src/resnet.py)                                              |  Supported | Doing | Doing | Doing | Doing
|Computer Vision (CV)  | Image Classification  | [ResNext50](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/resnext50/src/image_classification.py)                                             |  Supported |  Supported | Supported |  Supported | Doing
| Computer Vision (CV)  | Image Classification  | [VGG16](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/vgg16/src/vgg.py)                |  Supported |  Supported |  Supported |  Supported | Doing
| Computer Vision (CV)  | Image Classification  | [InceptionV3](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/inceptionv3/src/inception_v3.py)                |  Supported |  Supported |  Doing |  Doing | Doing
| Computer Vision (CV)  | Image Classification  | [DenseNet121](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/densenet121/src/network/densenet.py)                |  Supported |  Doing |  Doing |  Doing | Doing
| Computer Vision (CV)  | Mobile Image Classification<br>Image Classification<br>Semantic Segmentation  | [MobileNetV2](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/mobilenetv2/src/mobilenetV2.py)                   |  Supported |  Supported |  Supported |  Supported | Doing
| Computer Vision (CV)  | Mobile Image Classification<br>Image Classification<br>Semantic Segmentation  | [MobileNetV3](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/mobilenetv3/src/mobilenetV3.py)                   |  Doing |  Doing |  Supported |  Supported | Doing
|Computer Vision (CV)  | Object Detection  | [SSD](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/ssd/src/ssd.py)                   |  Supported |  Supported |Doing |Doing | Doing
| Computer Vision (CV)  | Object Detection  | [YoloV3-ResNet18](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/yolov3_resnet18/src/yolov3.py)         |  Supported |  Doing |  Doing |  Doing | Doing
| Computer Vision (CV)  | Object Detection  | [YoloV3-DarkNet53](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/yolov3_darknet53/src/yolo.py)         |  Supported |  Doing |  Doing |  Doing | Doing
| Computer Vision (CV)  | Object Detection  | [FasterRCNN](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/faster_rcnn/src/FasterRcnn/faster_rcnn_r50.py)         |  Supported |  Doing |  Doing |  Doing | Doing
| Computer Vision (CV)  | Object Detection  | [MaskRCNN](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/maskrcnn/src/maskrcnn/mask_rcnn_r50.py)         |  Supported |  Doing |  Doing |  Doing | Doing
| Computer Vision（CV） | Object Detection  | [WarpCTC](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/warpctc/src/warpctc.py)                    |  Supported |  Doing |  Supported |  Supported | Doing
| Computer Vision (CV) | Text Detection  | [PSENet](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/psenet/src/ETSNET/etsnet.py)                |  Supported |  Doing |  Doing |  Doing | Doing
| Computer Vision (CV) | Semantic Segmentation  | [DeeplabV3](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/deeplabv3/src/nets/deeplab_v3/deeplab_v3.py)                                           |  Supported |  Supported |  Doing |  Doing | Doing
| Computer Vision (CV)  | Semantic Segmentation  | [UNet2D-Medical](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/cv/unet/src/unet/unet_model.py)                |  Supported |  Doing |  Doing |  Doing | Doing
| Natural Language Processing (NLP) | Natural Language Understanding  | [BERT](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/nlp/bert/src/bert_model.py)                                          |  Supported |  Supported |  Supported |  Supported | Doing
| Natural Language Processing (NLP) | Natural Language Understanding  | [Transformer](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/nlp/transformer/src/transformer_model.py)                                          |  Supported |  Doing |  Doing |  Doing | Doing
| Natural Language Processing (NLP) | Natural Language Understanding  | [SentimentNet](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/nlp/lstm/src/lstm.py)                                          |  Doing |  Doing |  Supported |  Supported | Supported
| Natural Language Processing (NLP) | Natural Language Understanding  | [MASS](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/nlp/mass/src/transformer/transformer_for_train.py)                                          |  Supported |  Supported |  Doing |  Doing | Doing
| Natural Language Processing (NLP) | Natural Language Understanding  | [TinyBert](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/nlp/tinybert/src/tinybert_model.py)                                          |  Supported |  Doing |  Supported | Doing | Doing
| Recommender | Recommender System, CTR prediction  | [DeepFM](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/recommend/deepfm/src/deepfm.py)                                          |  Supported |  Supported |  Supported | Doing| Doing
| Recommender | Recommender System, Search ranking  | [Wide&Deep](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/recommend/wide_and_deep/src/wide_and_deep.py)                                          |  Supported |  Supported |  Supported | Supported | Doing
| Graph Neural Networks（GNN）| Text Classification  | [GCN](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/gnn/gcn/src/gcn.py)                                          |  Supported |  Doing |  Doing |  Doing | Doing
| Graph Neural Networks（GNN）| Text Classification  | [GAT](https://gitee.com/mindspore/mindspore/blob/r1.0/model_zoo/official/gnn/gat/src/gat.py)                                          |  Supported |  Doing |  Doing |  Doing | Doing

> You can also use [MindWizard Tool](https://gitee.com/mindspore/mindinsight/tree/r1.0/mindinsight/wizard/) to quickly generate classic network scripts.
