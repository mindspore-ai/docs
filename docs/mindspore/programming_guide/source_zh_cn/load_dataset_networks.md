# 典型网络数据加载和处理

`Ascend` `GPU` `CPU` `数据准备`

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/programming_guide/source_zh_cn/load_dataset_networks.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

|  领域 | 子领域  | 网络   |
|:----  |:-------  |:----   |
|计算机视觉 | 图像分类  | [AlexNet](https://gitee.com/mindspore/models/blob/master/official/cv/alexnet/src/dataset.py)
| 计算机视觉  | 图像分类  | [CNN](https://gitee.com/mindspore/models/blob/master/official/cv/cnn_direction_model/src/dataset.py)  |
| 计算机视觉  | 图像分类  | [GoogLeNet](https://gitee.com/mindspore/models/blob/master/official/cv/googlenet/src/dataset.py)   |
| 计算机视觉  | 图像分类  | [LeNet](https://gitee.com/mindspore/models/blob/master/official/cv/lenet/src/dataset.py)    |
| 计算机视觉  | 图像分类  | [MobileNetV3](https://gitee.com/mindspore/models/blob/master/official/cv/mobilenetv3/src/dataset.py)        |
| 计算机视觉  | 图像分类  | [ResNet-50](https://gitee.com/mindspore/models/blob/master/official/cv/resnet/src/dataset.py)   |
| 计算机视觉  | 图像分类  | [VGG16](https://gitee.com/mindspore/models/blob/master/official/cv/vgg16/src/dataset.py)  |
| 计算机视觉 | 目标检测  | [CenterFace](https://gitee.com/mindspore/models/blob/master/official/cv/centerface/src/dataset.py)     |
| 计算机视觉 | 目标检测  | [CTPN](https://gitee.com/mindspore/models/blob/master/official/cv/ctpn/src/dataset.py)     |
| 计算机视觉  | 目标检测  | [Faster R-CNN](https://gitee.com/mindspore/models/blob/master/official/cv/faster_rcnn/src/dataset.py)  |
| 计算机视觉  | 目标检测  | [Mask R-CNN](https://gitee.com/mindspore/models/blob/master/official/cv/maskrcnn/src/dataset.py)  |
| 计算机视觉  | 目标检测  | [SSD](https://gitee.com/mindspore/models/blob/master/official/cv/ssd/src/dataset.py) |
| 计算机视觉 | 目标检测  |[YOLOv4](https://gitee.com/mindspore/models/blob/master/official/cv/yolov4/src/yolo_dataset.py)         |
| 计算机视觉 | 文本检测  | [DeepText](https://gitee.com/mindspore/models/blob/master/official/cv/deeptext/src/dataset.py)                |
| 计算机视觉 | 语义分割  | [DeepLabV3](https://gitee.com/mindspore/models/blob/master/official/cv/deeplabv3/src/data/dataset.py)   |
| 计算机视觉 | 关键点检测  |[OpenPose](https://gitee.com/mindspore/models/blob/master/official/cv/openpose/src/dataset.py)                |
| 计算机视觉 | 关键点检测  |[SimplePoseNet](https://gitee.com/mindspore/models/blob/master/official/cv/simple_pose/src/dataset.py)                |
| 计算机视觉 | 光学字符识  |[CRNN](https://gitee.com/mindspore/models/blob/master/official/cv/crnn/src/dataset.py)                |
| 自然语言处理 | 自然语言理解  | [BERT](https://gitee.com/mindspore/models/blob/master/official/nlp/bert/src/dataset.py)  |
| 自然语言处理 | 自然语言理解  | [FastText](https://gitee.com/mindspore/models/blob/master/official/nlp/fasttext/src/dataset.py)    |
| 自然语言处理 | 自然语言理解  | [GRU](https://gitee.com/mindspore/models/blob/master/official/nlp/gru/src/dataset.py)            |
| 自然语言处理 | 自然语言理解  | [Transformer](https://gitee.com/mindspore/models/blob/master/official/nlp/transformer/src/dataset.py)  |
| 自然语言处理 | 自然语言理解  | [TinyBERT](https://gitee.com/mindspore/models/blob/master/official/nlp/tinybert/src/dataset.py)   |
| 自然语言处理 | 自然语言理解  | [TextCNN](https://gitee.com/mindspore/models/blob/master/official/nlp/textcnn/src/dataset.py)            |
| 推荐 | 推荐系统、点击率预估  | [DeepFM](https://gitee.com/mindspore/models/blob/master/official/recommend/deepfm/src/dataset.py)    |
| 推荐 | 推荐系统、搜索、排序  | [Wide&Deep](https://gitee.com/mindspore/models/blob/master/official/recommend/wide_and_deep/src/datasets.py)      |
| 推荐 | 推荐系统  | [NAML](https://gitee.com/mindspore/models/blob/master/official/recommend/naml/src/dataset.py)             |
| 推荐 | 推荐系统  | [NCF](https://gitee.com/mindspore/models/blob/master/official/recommend/ncf/src/dataset.py)    |
| 图神经网络 | 文本分类  | [GCN](https://gitee.com/mindspore/models/blob/master/official/gnn/gcn/src/dataset.py)  |
| 图神经网络 | 文本分类  | [GAT](https://gitee.com/mindspore/models/blob/master/official/gnn/gat/src/dataset.py) |
| 图神经网络 | 推荐系统 | [BGCF](https://gitee.com/mindspore/models/blob/master/official/gnn/bgcf/src/dataset.py) |
