# Glossary

<a href="https://gitee.com/mindspore/docs/blob/r0.6/docs/source_en/glossary.md" target="_blank"><img src="./_static/logo_source.png"></a>

|  Acronym and Abbreviation  |  Description  | 
| -----    | -----    |
| ACL | Ascend Computer Language, for users to develop deep neural network applications, which provides the C++ API library including device management, context management, stream management, memory management, model loading and execution, operator loading and execution, media data processing, etc. |
|  Ascend  |  Name of Huawei Ascend series chips.  |
|  CCE  | Cube-based Computing Engine, which is an operator development tool oriented to hardware architecture programming.  |
|  CCE-C  |  Cube-based Computing Engine C, which is C code developed by the CCE.  |
|  CheckPoint  |  MindSpore model training check point, which is used to save model parameters for inference or retraining.  |
|  CIFAR-10  |  An open-source image dataset that contains 60000 32 x 32 color images of 10 categories, with 6000 images of each category. There are 50000 training images and 10000 test images.  |
|  CIFAR-100  |  An open-source image dataset that contains 100 categories. Each category has 500 training images and 100 test images.  |
|  DaVinci  |  DaVinci architecture, Huawei-developed new chip architecture.  |
|  EulerOS  |  Euler operating system, which is developed by Huawei based on the standard Linux kernel.  |
|  FC Layer  |  Fully connected layer, which acts as a classifier in the entire convolutional neural network.  |
|  FE  |  Fusion Engine, which connects to GE and TBE operators and has the capabilities of loading and managing the operator information library and managing convergence rules.  |
|  Fine-tuning |  A process to take a network model that has already been trained for a given task, and make it perform a second similar task.  |
|  FP16  |  16-bit floating point, which is a half-precision floating point arithmetic format, consuming less memory.  |
|  FP32  |  32-bit floating point, which is a single-precision floating point arithmetic format.  |
|  GE  |  Graph Engine, MindSpore computational graph execution engine, which is responsible for optimizing hardware (such as operator fusion and memory overcommitment) based on the front-end computational graph and starting tasks on the device side.  |
| GEIR | Graph Engine Intermediate Representation, such as ONNX, it is an open file format for machine learning. It is defined by Huawei and is better suited to Ascend AI processor.|
|  GHLO  |  Graph High Level Optimization. GHLO includes optimization irrelevant to hardware (such as dead code elimination), auto parallel, and auto differentiation.  |
|  GLLO  |  Graph Low Level Optimization. GLLO includes hardware-related optimization and in-depth optimization related to the combination of hardware and software, such as operator fusion and buffer fusion.  |
|  Graph Mode  |  MindSpore static graph mode. In this mode, the neural network model is compiled into an entire graph and then delivered for execution, featuring high performance.  |
|  HCCL  |  Huawei Collective Communication Library, which implements multi-device and multi-card communication based on the Da Vinci architecture chip.  |
|  ImageNet  |  Image database organized based on the WordNet hierarchy (currently nouns only).  |
|  LeNet  |  A classical convolutional neural network architecture proposed by Yann LeCun and others.  |
|  Loss  |  Difference between the predicted value and the actual value, which is a standard for determining the model quality of deep learning.  |
|  LSTM  |  Long short-term memory, an artificial recurrent neural network (RNN) architecture used for processing and predicting an important event with a long interval and delay in a time sequence.  |
|  Manifest  |  A data format file. Huawei ModelArt adopts this format. For details, see <https://support.huaweicloud.com/engineers-modelarts/modelarts_23_0009.html>.  |
|  ME  |  Mind Expression, MindSpore frontend, which is used to compile tasks from user source code to computational graphs, control execution during training, maintain contexts (in non-sink mode), and dynamically generate graphs (in PyNative mode).  |
|  MindArmour  |  MindSpore security component, which is used for AI adversarial example management, AI model attack defense and enhancement, and AI model robustness evaluation.  |
|  MindData  |  MindSpore data framework, which provides data loading, enhancement, dataset management, and visualization.  |
|  MindInsight  |  MindSpore visualization component, which visualizes information such as scalars, images, computational graphs, and model hyperparameters.  |
|  MindSpore  |  Huawei-leaded open-source deep learning framework.  |
|  MindSpore Predict  |  A lightweight deep neural network inference engine that provides the inference function for models trained by MindSpore on the device side.  |
|  MNIST database  |  Modified National Handwriting of Images and Technology database, a large handwritten digit database, which is usually used to train various image processing systems.  |
| ONNX | Open Neural Network Exchange, is an open format built to represent machine learning models.|
|  PyNative Mode  |  MindSpore dynamic graph mode. In this mode, operators in the neural network are delivered and executed one by one, facilitating the compilation and debugging of the neural network model.  |
|  ResNet-50  |  Residual Neural Network 50, a residual neural network proposed by four Chinese people, including Kaiming He from Microsoft Research Institute.  |
|  Schema  |  Data set structure definition file, which defines the fields contained in a dataset and the field types.  |
|  Summary  |  An operator that monitors the values of tensors on the network. It is a peripheral operation in the figure and does not affect the data flow.  |
|  TBE  |  Tensor Boost Engine, an operator development tool that is extended based on the Tensor Virtual Machine (TVM) framework.  |
|  TFRecord  |  Data format defined by TensorFlow.  |
