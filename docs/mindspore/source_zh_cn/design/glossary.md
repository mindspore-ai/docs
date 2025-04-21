# 术语

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_zh_cn/design/glossary.md)

|  术语/缩略语  |  说明  |
| -----    | -----    |
|  ACL  | Ascend Computer Language，提供Device管理、Context管理、Stream管理、内存管理、模型加载与执行、算子加载与执行、媒体数据处理等C++ API库，供用户开发深度神经网络应用。|
|  Ascend  |  华为昇腾系列芯片的系列名称。  |
|  Backpropagation  |  反向传播。  |
|  Batch  |  模型训练的一次迭代（即一次梯度更新）中使用的样本集。  |
|  Batch size  |  模型迭代一次，使用的样本集的大小。  |
|  CCE  |  Cube-based Computing Engine，面向硬件架构编程的算子开发工具。  |
|  CCE-C  |  Cube-based Computing Engine C，使用CCE开发的C代码。  |
|  CheckPoint  |  MindSpore模型训练检查点，保存模型的参数，可以用于保存模型供推理，或者再训练。  |
|  CIFAR-10  |  一个开源的图像数据集，包含10个类别的60000个32x32彩色图像，每个类别6000个图像。有50000张训练图像和10000张测试图像。  |
|  CIFAR-100  |  一个开源的图像数据集，它有100个类别，每个类别包含500张训练图像和100张测试图像。  |
|  Clip  |  梯度裁剪。  |
|  Davinci  |  达芬奇架构，华为自研的新型芯片架构。  |
|  Device  |  设备侧，主要执行MindSpore算子的硬件，包括Ascend、GPU、CPU等。  |
|  Device_id  |  分布式并行中，卡的物理ID。  |
|  Dimension Reduction  |  降维。  |
|  Epoch  |  数据集的一次完整遍历。 |
|  EulerOS  |  欧拉操作系统，华为自研的基于Linux标准内核的操作系统。  |
|  FC Layer  |  Fully Conneted Layer，全连接层。整个卷积神经网络中起到分类器的作用。  |
|  FE  |  Fusion Engine，负责对接GE和TBE算子，具备算子信息库的加载与管理、融合规则管理等能力。  |
|  Fine-tuning |  基于面向某任务训练的网络模型，训练面向第二个类似任务的网络模型。  |
|  Format  |  数据格式，如NCHW、NHWC、NC0HWC1等。N: batch size; C: channel; H: height; W:width。<br>昇腾AI软件栈中，张量数据统一采用NC0HWC1的五维数据格式。其中C0等于AI Core中矩阵计算单元的大小，对于FP16类型为16，对于INT8类型则为32，这部分数据需要连续存储；C1是将C维度按照C0进行拆分后的数目，即C1=C/C0。如果结果不整除，最后一份数据需要补零以对齐C0。  |
|  FP16  |  16位浮点，半精度浮点算术，消耗更小内存。  |
|  FP32  |  32位浮点，单精度浮点算术。  |
|  GE  |  Graph Engine，MindSpore计算图执行引擎，主要负责根据前端的计算图完成硬件相关的优化（算子融合、内存复用等等）、device侧任务启动。  |
|  GHLO  |  Graph High Level Optimization，计算图高级别优化。GHLO包含硬件无关的优化（如死代码消除等）、自动并行和自动微分等功能。  |
|  GLLO  |  Graph Low Level Optimization，计算图低级别优化。GLLO包含硬件相关的优化，以及算子融合、Buffer融合等软硬件结合相关的深度优化。  |
|  Graph Mode  |  MindSpore的静态图模式，将神经网络模型编译成一整张图，然后下发执行，性能高。  |
|  HCCL  |  Huawei Collective Communication Library，实现了基于Davinci架构芯片的多机多卡通信。  |
|  Host  |  主机侧，主要进行图编译、数据处理等。  |
|  ImageNet  |  根据WordNet层次结构（目前仅名词）组织的图像数据库。  |
|  Layout  |  分布式并行中，数据在卡上的分布情况。  |
|  LeNet  |  一个经典的卷积神经网络架构，由Yann LeCun等人提出。  |
|  Loss  |  损失，预测值与实际值的偏差，深度学习用于判断模型好坏的一个标准。  |
|  Loss scale  |  为了防止梯度下溢而进行的梯度放大。  |
|  LSTM  |  Long short-term memory，长短期记忆，对应的网络是一种时间循环神经网络，适合于处理和预测时间序列中间隔和延迟非常长的重要事件。  |
|  Manifest  |  一种数据格式文件，华为ModelArts采用了该格式，详见[导入Manifest文件的规范说明](https://support.huaweicloud.com/intl/zh-cn/dataprepare-modelarts/dataprepare-modelarts-0015.html)。  |
|  ME  |  Mind Expression，MindSpore前端，主要完成从用户源码到计算图的编译任务、训练中控制执行及上下文维护（非下沉模式配置下）、动态图（PyNative模式）等。  |
|  MindSpore Armour  |  MindSpore安全模块，通过差分隐私、对抗性攻防等技术手段，提升模型的保密性、完整性和可用性，阻止攻击者对模型进行恶意修改或是破解模型的内部构件，窃取模型的参数。  |
|  MindData  |  MindSpore数据框架，提供数据加载、增强、数据集管理以及可视化。  |
|  MindIR  |  MindSpore IR，一种基于图表示的函数式IR，定义了可扩展的图结构以及算子IR表示，存储了MindSpore基础数据结构。 |
|  MindRecord  |  MindSpore定义的一种数据格式，是一个执行读取、写入、搜索和转换MindSpore格式数据集的模块。  |
|  MindSpore  |  华为主导开源的深度学习框架。  |
|  MindSpore Lite  |  一个轻量级的深度神经网络推理引擎，提供了将MindSpore训练出的模型在端侧进行推理的功能。  |
|  MNIST database  |  Modified National Institute of Standards and Technology database，一个大型手写数字数据库，通常用于训练各种图像处理系统。  |
|  NCCL  |  Nvidia Collective multi-GPU Communication Library的简称，它是一个实现多GPU的collective communication通信库。  |
|  ONNX  | Open Neural Network Exchange，是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。|
|  PyNative Mode  |  MindSpore的动态图模式，将神经网络中的各个算子逐一下发执行，方便用户编写和调试神经网络模型。  |
|  Proto  |  ProtoBuffer文件格式。  |
|  ResNet-50  |  Residual Neural Network 50，由微软研究院的Kaiming He等四名华人提出的残差神经网络。  |
|  Rank_id  |  分布式并行中，卡的逻辑ID。  |
|  Scalar  |  标量Tensor，其shape为（）。  |
|  Schema  |  数据集结构定义文件，用于定义数据集包含哪些字段以及字段的类型。  |
|  Shape  |  张量在各种维度中包含的元素数，如Tensor[2,3]，shape为{2,3}，表示是一个二维张量，第一维有2行，第二维有3列，共2*3=6个元素。  |
|  Step或Iteration  |  完成一次前向计算和反向传播。  |
|  Summary  |  是对网络中Tensor取值进行监测的一种算子，在图中是“外围”操作，不影响数据流本身。  |
|  TBE  |  Tensor Boost Engine，华为自研的NPU算子开发工具，在TVM（ Tensor Virtual Machine ）框架基础上扩展，提供了一套Python API来实施开发活动，进行自定义算子开发。 |
|  TFRecord  |  Tensorflow定义的数据格式。  |
|  Tensor  |  张量，存储多维数组的数据结构。最常见的是标量、向量或矩阵。  |
|  广播  |  在矩阵数学运算中，是将操作数的shape扩展到与该运算兼容的维。在分布式并行中，是某卡上的参数同步到其他卡上。  |
|  计算图下沉  | 计算图整图下沉到Device上执行，减少Host-Device交互开销。详见[面向昇腾硬件的竞争力优化](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/design/overview.html#面向昇腾硬件的竞争力优化)。 |
|  循环下沉  |  在On Device执行的基础上的优化，目的是进一步减少Host侧和Device侧之间的交互次数。详见[面向昇腾硬件的竞争力优化](https://www.mindspore.cn/docs/zh-CN/r2.6.0rc1/design/overview.html#面向昇腾硬件的竞争力优化)。  |
|  数据下沉  |  数据通过通道直接传送到Device上。  |
|  图模式  |  又称静态图模式，将神经网络模型编译成一整张图，然后下发执行。该模式利用图优化等技术提高运行性能，同时有助于规模部署和跨平台运行。  |
|  PyNative模式  |  动态图模式，将神经网络中的各个算子逐一下发执行，方便用户编写和调试神经网络模型。  |
