# 术语

`Ascend` `GPU` `CPU` `设计`

<a href="https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/programming_guide/source_zh_cn/design/glossary.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source.png"></a>

|  术语/缩略语  |  说明  |
| -----    | -----    |
|  ACL  | Ascend Computer Language，提供Device管理、Context管理、Stream管理、内存管理、模型加载与执行、算子加载与执行、媒体数据处理等C++ API库，供用户开发深度神经网络应用。|
|  AIR  |  Ascend Intermediate Representation，类似ONNX，是华为定义的针对机器学习所设计的开放式的文件格式，能更好地适配Ascend AI处理器。 |
|  Ascend  |  华为昇腾系列芯片的系列名称。  |
|  CCE  |  Cube-based Computing Engine，面向硬件架构编程的算子开发工具。  |
|  CCE-C  |  Cube-based Computing Engine C，使用CCE开发的C代码。  |
|  CheckPoint  |  MindSpore模型训练检查点，保存模型的参数，可以用于保存模型供推理，或者再训练。  |
|  CIFAR-10  |  一个开源的图像数据集，包含10个类别的60000个32x32彩色图像，每个类别6000个图像。有50000张训练图像和10000张测试图像。  |
|  CIFAR-100  |  一个开源的图像数据集，它有100个类别，每个类别包含500张训练图像和100张测试图像。  |
|  Davinci  |  达芬奇架构，华为自研的新型芯片架构。  |
|  EulerOS  |  欧拉操作系统，华为自研的基于Linux标准内核的操作系统。  |
|  FC Layer  |  Fully Conneted Layer，全连接层。整个卷积神经网络中起到分类器的作用。  |
|  FE  |  Fusion Engine，负责对接GE和TBE算子，具备算子信息库的加载与管理、融合规则管理等能力。  |
|  Fine-tuning |  基于面向某任务训练的网络模型，训练面向第二个类似任务的网络模型。  |
|  FP16  |  16位浮点，半精度浮点算术，消耗更小内存。  |
|  FP32  |  32位浮点，单精度浮点算术。  |
|  GE  |  Graph Engine，MindSpore计算图执行引擎，主要负责根据前端的计算图完成硬件相关的优化（算子融合、内存复用等等）、device侧任务启动。  |
|  GHLO  |  Graph High Level Optimization，计算图高级别优化。GHLO包含硬件无关的优化（如死代码消除等）、自动并行和自动微分等功能。  |
|  GLLO  |  Graph Low Level Optimization，计算图低级别优化。GLLO包含硬件相关的优化，以及算子融合、Buffer融合等软硬件结合相关的深度优化。  |
|  Graph Mode  |  MindSpore的静态图模式，将神经网络模型编译成一整张图，然后下发执行，性能高。  |
|  HCCL  |  Huawei Collective Communication Library，实现了基于Davinci架构芯片的多机多卡通信。  |
|  ImageNet  |  根据WordNet层次结构（目前仅名词）组织的图像数据库。  |
|  LeNet  |  一个经典的卷积神经网络架构，由Yann LeCun等人提出。  |
|  Loss  |  损失，预测值与实际值的偏差，深度学习用于判断模型好坏的一个标准。  |
|  LSTM  |  Long short-term memory，长短期记忆，对应的网络是一种时间循环神经网络，适合于处理和预测时间序列中间隔和延迟非常长的重要事件。  |
|  Manifest  |  一种数据格式文件，华为ModelArts采用了该格式，详细说明请参见<https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0009.html>。  |
|  ME  |  Mind Expression，MindSpore前端，主要完成从用户源码到计算图的编译任务、训练中控制执行及上下文维护（非下沉模式配置下）、动态图（PyNative模式）等。  |
|  MindArmour  |  MindSpore安全模块，通过差分隐私、对抗性攻防等技术手段，提升模型的保密性、完整性和可用性，阻止攻击者对模型进行恶意修改或是破解模型的内部构件，窃取模型的参数。  |
|  MindData  |  MindSpore数据框架，提供数据加载、增强、数据集管理以及可视化。  |
|  MindInsight  |  MindSpore可视化组件，可视化标量、图像、计算图以及模型超参等信息。  |
|  MindIR  |  MindSpore IR，一种基于图表示的函数式IR，定义了可扩展的图结构以及算子IR表示，存储了MindSpore基础数据结构。 |
|  MindRecord  |  MindSpore定义的一种数据格式，是一个执行读取、写入、搜索和转换MindSpore格式数据集的模块。  |
|  MindSpore  |  华为主导开源的深度学习框架。  |
|  MindSpore Lite  |  一个轻量级的深度神经网络推理引擎，提供了将MindSpore训练出的模型在端侧进行推理的功能。  |
|  MNIST database  |  Modified National Institute of Standards and Technology database，一个大型手写数字数据库，通常用于训练各种图像处理系统。  |
|  ONNX  | Open Neural Network Exchange，是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。|
|  PyNative Mode  |  MindSpore的动态图模式，将神经网络中的各个算子逐一下发执行，方便用户编写和调试神经网络模型。  |
|  ResNet-50  |  Residual Neural Network 50，由微软研究院的Kaiming He等四名华人提出的残差神经网络。  |
|  Schema  |  数据集结构定义文件，用于定义数据集包含哪些字段以及字段的类型。  |
|  Summary  |  是对网络中Tensor取值进行监测的一种算子，在图中是“外围”操作，不影响数据流本身。  |
|  TBE  |  Tensor Boost Engine，华为自研的NPU算子开发工具，在TVM（ Tensor Virtual Machine ）框架基础上扩展，提供了一套Python API来实施开发活动，进行自定义算子开发。 |
|  TFRecord  |  Tensorflow定义的数据格式。  |
