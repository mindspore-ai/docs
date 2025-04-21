# Glossary

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/docs/mindspore/source_en/design/glossary.md)

|  Acronym and Abbreviation  |  Description  |
| -----    | -----    |
|  ACL  |  Ascend Computer Language, for users to develop deep neural network applications, which provides the C++ API library including device management, context management, stream management, memory management, model loading and execution, operator loading and execution, media data processing, etc. |
|  Ascend  |  Name of Huawei Ascend series chips.  |
|  Backpropagation  |  Backpropagation, short for "backward propagation of errors"  |
|  Batch  |  Batch means a group of training samples. |
|  Batch size  |  The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.  |
|  CCE  | Cube-based Computing Engine, which is an operator development tool oriented to hardware architecture programming.  |
|  CCE-C  |  Cube-based Computing Engine C, which is C code developed by the CCE.  |
|  CheckPoint  |  MindSpore model training check point, which is used to save model parameters for inference or retraining.  |
|  CIFAR-10  |  An open-source image dataset that contains 60000 32 x 32 color images of 10 categories, with 6000 images of each category. There are 50000 training images and 10000 test images.  |
|  CIFAR-100  |  An open-source image dataset that contains 100 categories. Each category has 500 training images and 100 test images.  |
|  Clip  |   Gradient clipping.  |
|  DaVinci  |  DaVinci architecture, Huawei-developed new chip architecture.  |
|  Device  |  Hardware for executing MindSpore operators, including Ascend, GPU, CPU, etc.  |
|  Device_id  |  The identification of devices in a distributed environment.  |
|  Dimension Reduction  |  Dimension Reduction is the transformation of data from a high-dimensional space into a low-dimensional space.  |
|  Epoch  |  The learning algorithm will work through the entire training dataset.  |
|  EulerOS  |  Euler operating system, which is developed by Huawei based on the standard Linux kernel.  |
|  FC Layer  |  Fully connected layer, which acts as a classifier in the entire convolutional neural network.  |
|  FE  |  Fusion Engine, which connects to GE and TBE operators and has the capabilities of loading and managing the operator information library and managing convergence rules.  |
|  Fine-tuning |  A process to take a network model that has already been trained for a given task, and make it perform a second similar task.  |
|  Format  |   Data format, such as NCHW, NHWA, NC0HWC1, and so on, where N denotes the batch size, C denotes channel, H denotes height, and W denotes width. <br> In the Ascend environment, when using stacks, the tensor format requires a 5-D tensor in the format of NC0HWC1, where C0 represents the size of the matrix calculation unit in AI Core. If the data type is FP16, then C0 equals to 16. If the data type is INT8, then C0 equals to 32. These data should be stored continuously. The value of C1 is obtained after the C dimension is split according to C0, i.e., C1=C/C0. If If the result is not divisible, the last data needs to be zero-padded to align with C0.  |
|  FP16  |  16-bit floating point, which is a half-precision floating point arithmetic format, consuming less memory.  |
|  FP32  |  32-bit floating point, which is a single-precision floating point arithmetic format.  |
|  GE  |  Graph Engine, MindSpore computational graph execution engine, which is responsible for optimizing hardware (such as operator fusion and memory overcommitment) based on the front-end computational graph and starting tasks on the device side.  |
|  GHLO  |  Graph High Level Optimization. GHLO includes optimization irrelevant to hardware (such as dead code elimination), auto parallel, and auto differentiation.  |
|  GLLO  |  Graph Low Level Optimization. GLLO includes hardware-related optimization and in-depth optimization related to the combination of hardware and software, such as operator fusion and buffer fusion.  |
|  Graph Mode  |  MindSpore static graph mode. In this mode, the neural network model is compiled into an entire graph and then delivered for execution, featuring high performance.  |
|  HCCL  |  Huawei Collective Communication Library, which implements multi-device and multi-card communication based on the Da Vinci architecture chip.  |
|  Host  |  Host side. It is used for graph compilation, data processing, etc.  |
|  ImageNet  |  Image database organized based on the WordNet hierarchy (currently nouns only).  |
|  Layout  |  The distribution of data on the card in a distributed parallel environment.  |
|  LeNet  |  A classical convolutional neural network architecture proposed by Yann LeCun and others.  |
|  Loss  |  Difference between the predicted value and the actual value, which is a standard for determining the model quality of deep learning.  |
|  Loss scale  |  Gradient amplification to prevent gradient underflow.  |
|  LSTM  |  Long short-term memory, an artificial recurrent neural network (RNN) architecture used for processing and predicting an important event with a long interval and delay in a time sequence.  |
|  Manifest  |  A data format file. Huawei ModelArt adopts this format. For details, see [Specification of Manifest](https://support.huaweicloud.com/intl/en-us/dataprepare-modelarts/dataprepare-modelarts-0015.html).  |
|  ME  |  Mind Expression, MindSpore frontend, which is used to compile tasks from user source code to computational graphs, control execution during training, maintain contexts (in non-sink mode), and dynamically generate graphs (in PyNative mode).  |
|  MindSpore Armour  |  The security module of MindSpore, which improves the confidentiality, integrity and usability of the model through technical means such as differential privacy and adversarial attack and defense. MindSpore Armour prevents attackers from maliciously modifying the model or cracking the internal components of the model to steal the parameters of the model.  |
|  MindData  |  MindSpore data framework, which provides data loading, enhancement, dataset management, and visualization.  |
|  MindIR  |  MindSpore IR, a functional IR based on graph representation, defines a scalable graph structure and operator IR representation, and stores the basic data structure of MindSpore.  |
|  MindRecord  |  It is a data format defined by MindSpore, it is a module for reading, writing, searching and converting datasets in MindSpore format.  |
|  MindSpore  |  Huawei-leaded open-source deep learning framework.  |
|  MindSpore Lite  |  A lightweight deep neural network inference engine that provides the inference function for models trained by MindSpore on the device side.  |
|  MNIST database  |  Modified National Handwriting of Images and Technology database, a large handwritten digit database, which is usually used to train various image processing systems.  |
|  NCCL  |  Short for Nvidia Collective multi-GPU Communication Library. It is a collective communication library for communicating multiple GPUs.  |
|  ONNX  |  Open Neural Network Exchange, is an open format built to represent machine learning models.|
|  PyNative Mode  |  MindSpore dynamic graph mode. In this mode, operators in the neural network are delivered and executed one by one, facilitating the compilation and debugging of the neural network model.  |
|  Proto  |  Short for ProtoBuffer data format.  |
|  ResNet-50  |  Residual Neural Network 50, a residual neural network proposed by four Chinese people, including Kaiming He from Microsoft Research Institute.  |
|  Rank_id  |  The logic ID of cards in a distributed parallel environment.  |
|  Scalar  |   Scalars are single numbers and are an example of a 0th-order tensor with certain shapes.  |
|  Schema  |  Dataset structure definition file, which defines the fields contained in a dataset and the field types.  |
|  Shape  |   The dimension of tensors, such as Tensor[2,3], it has the shape of {2,3}, indicating that it is a 2D tensor. The 1st dimension has 2 rows, while the 2nd dimension has 3 columns. In total, there are 2*3=6 elements.  |
|  Step or Iterationn |  Complete a forward calculation and backpropagation. |
|  Summary  |  An operator that monitors the values of tensors on the network. It is a peripheral operation in the figure and does not affect the data flow.  |
|  TBE  |  Tensor Boost Engine, it is a self-developed NPU operator development tool developed by Huawei, which is expanded on the basis of the TVM (Tensor Virtual Machine) framework. It provides a set of Python API to implement development activities and develop custom operators.   |
|  TFRecord  |  Data format defined by TensorFlow.  |
|  Tensor  |  A tensor is a generalization of vectors and matrices and is easily understood as a multidimensional array, scalar, and matrix.  |
|  Broadcast  |   In matrix mathematical operations, the shape of the operands is extended to a dimension compatible with the operation. In distributed parallelism, the parameters on one card are synchronized to other cards.  |
|  Computational Graphs on Devices  | The entire graph is executed on the device to reduce the interaction overheads between the host and device. |
|  Cyclic Sinking  |  Cyclic sinking is optimized based on on-device execution to further reduce the number of interactions between the host and device. Refer to [Competitive Optimization for Ascend Hardware](https://www.mindspore.cn/docs/en/r2.6.0rc1/design/overview.html#competitive-optimization-for-ascend-hardware) for more details. |
|  Data Sinking |  Sinking means that data is directly transmitted to the device through a channel. Refer to [Competitive Optimization for Ascend Hardware](https://www.mindspore.cn/docs/en/r2.6.0rc1/design/overview.html#competitive-optimization-for-ascend-hardware) for more details. |
|  Graph Mode |   Static graph mode or graph mode. In this mode, the neural network model is compiled into an entire graph, and then the graph is delivered for execution. This mode uses graph optimization to improve the running performance and facilitates large-scale deployment and cross-platform running.  |
|  PyNative Mode  |  Dynamic graph mode. In this mode, operators in the neural network are delivered and executed one by one, facilitating the compilation and debugging of the neural network model.  |
