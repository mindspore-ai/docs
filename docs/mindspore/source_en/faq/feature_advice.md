# Feature Advice

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/faq/feature_advice.md)

## Q: Is the `input=np.random.uniform(...)` format fixed when the MindIR format is exported?

A: The format is not fixed. This step is to create an input for constructing the network structure. You only need to input the correct `shape` in `export`. You can use `np.ones` and `np.zeros` to create an input.

<br/>

## Q: What framework models and formats can be directly read by MindSpore? Can the PTH Model obtained through training in PyTorch be loaded to the MindSpore framework for use?

A: MindSpore uses Protobuf to store training parameters and cannot directly read framework models. A model file stores parameters and their values. You can use APIs of other frameworks to read parameters, obtain the key-value pairs of parameters, and load the key-value pairs to MindSpore. If you want to use the .ckpt file trained by other framework, read the parameters and then call the [save_checkpoint](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.save_checkpoint.html) API of MindSpore to save the file as a .ckpt file that can be read by MindSpore.

<br/>

## Q: What is the difference between the PyNative and Graph modes?

A: Compare through the following four aspects:

- In terms of network execution: operators used in the two modes are the same. Therefore, when the same network and operators are executed in the two modes, the accuracy is the same. As Graph mode uses graph optimization, calculation graph sinking and other technologies, it has higher performance and efficiency in executing the network.

- In terms of application scenarios: Graph mode requires the network structure to be built at the beginning, and then the framework performs entire graph optimization and execution. This mode is suitable to scenarios where the network is fixed and high performance is required.

- In term of different hardware (such as `Ascend`, `GPU`, and `CPU`) resources: the two modes are supported.

- In terms of code debugging: since operators are executed line by line in PyNative mode, you can directly debug the Python code and view the `/api` output or execution result of the corresponding operator at any breakpoint in the code. In Graph mode, the network is built but not executed in the constructor function. Therefore, you cannot obtain the output of the corresponding operator at breakpoints in the `construct` function. You can only specify operators and print their output results, and then view the results after the network execution is completed.

- In terms of syntax support: PyNative mode has dynamic syntax affinity, flexible expression, and convers almost all Python syntax. Graph mode supports a subset of common Python syntax to support the construction and training of neural networks.

<br/>

## Q: Does MindSpore run only on Huawei `Ascend`?

A: MindSpore supports Huawei `Ascend`, `GPUs`, and `CPUs`, and supports heterogeneous computing.

<br/>

## Q: If MindSpore and PyTorch are installed in an environment, can the syntax of the two frameworks be used together in a Python file?

A: You can use the two frameworks in a python file. Pay attention to the differences between types. For example, the tensor types created by the two frameworks are different, but the basic types of Python are general.

<br/>

## Q: Can MindSpore read a ckpt file of TensorFlow?

A: The formats of  `ckpt` of MindSpore and `ckpt`of TensorFlow are not generic. Although both use the `Protocol` Buffers, the definition of `proto` are different. Currently, MindSpore cannot read the TensorFlow or Pytorch `ckpt` files.

<br/>

## Q: How do I use models trained by MindSpore on Atlas 200/300/500 inference product? Can they be converted to models used by HiLens Kit?

A: Yes. HiLens Kit uses Atlas 200/300/500 inference product as the inference core. Therefore, the two questions are essentially the same, which both need to convert as OM model. Atlas 200/300/500 inference product requires a dedicated OM model. Use MindSpore to export the ONNX and convert it into an OM model supported by Atlas 200/300/500 inference product. For details, see [Multi-platform Inference](https://www.mindspore.cn/tutorials/en/master/model_infer/ms_infer/llm_inference_overview.html).

<br/>

## Q: Does MindSpore only be run on Huawei own `Ascend`?

A: MindSpore supports Huawei's own `Ascend` in addition to `GPU` and `CPU`, which is support for heterogeneous computing power.

<br/>

## Q: Does MindSpore have any limitation on the input size of a single Tensor for exporting and loading models?

A: Due to hardware limitations of Protobuf, when exporting to ONNX formats, the size of model parameters cannot exceed 2G; when exporting to MINDIR format, there is no limit to the size of model parameters. MindSpore only supports the importing of MINDIR and doesn't support the importing of ONNX formats. The importing of MINDIR does not have size limitation.

<br/>

## Q: Does MindSpore have any plan on supporting heterogeneous computing hardwares?

A: MindSpore provides pluggable device management interface, so that developer could easily integrate other types of heterogeneous computing hardwares (like FPGA) to MindSpore. We welcome more backend support in MindSpore from the community.

<br/>

## Q: What is the relationship between MindSpore and ModelArts? Can we use MindSpore in ModelArts?

A: ModelArts is Huawei public cloud online training and inference platform, and MindSpore is Huawei AI framework.

<br/>

## Q: The recent announced programming language such as taichi got Python extensions that could be directly used as `import taichi as ti`. Does MindSpore have similar support?

A: MindSpore supports Python native expression and `import mindspore` related package can be used.

<br/>

## Q: Does MindSpore support truncated gradient?

A: Yes. For details, see [Definition and Usage of Truncated Gradient](https://gitee.com/mindspore/models/blob/master/official/nlp/Transformer/src/transformer_for_train.py#L35).

<br/>

## Q: What is the MindSpore IR design concept?

A: Function expression: All expressions are functions, and differentiation and automatic parallel analysis are easy to implement without side effect. `JIT` compilation capability: The graph-based IR, control flow dependency, and data flow are combined to balance the universality and usability. Graphically complete IR: More conversion `Python` flexible syntax, including recursion, etc.

<br/>

## Q: What are the advantages and features of MindSpore parallel model training?

A: In addition to data parallelism, MindSpore distributed training also supports operator-level model parallelism. The operator input tensor can be tiled and parallelized. On this basis, automatic parallelism is supported. You only need to write a single-device script to automatically tile the script to multiple nodes for parallel execution.

<br/>

## Q: How does MindSpore implement semantic collaboration and processing? Is the popular Formal Concept Analysis (FCA) used?

A: The MindSpore framework does not support FCA. For semantic models, you can call third-party tools to perform FCA in the data preprocessing phase. MindSpore supports Python therefore `import FCA` related package could do the trick.

<br/>

## Q: Does MindSpore have any plan on the edge and device when the training and inference functions of MindSpore on the cloud are relatively mature?

A: MindSpore is a unified cloud-edge-device training and inference framework, which supports exporting cloud-side trained models to Ascend AI processors and terminal devices for inference. The optimizations supported in the current inference stage include quantization, operator fusion, and memory overcommitment.

<br/>

## Q: How does MindSpore support automatic parallelism?

A: Automatic parallelism on CPUs and GPUs are being improved. You are advised to use the automatic parallelism on the Atlas training series. Follow our open source community and apply for a MindSpore developer experience environment for trial use.

<br/>

## Q: Does MindSpore have a similar module that can implement object detection algorithms based on TensorFlow?

A: The TensorFlow's object detection Pipeline API belongs to the TensorFlow's Model module. After MindSpore's detection models are complete, similar Pipeline APIs will be provided.

<br/>

## Q: How do I perform transfer learning in PyNative mode?

A: PyNative mode is compatible with transfer learning.

<br/>

## Q: What is the relationship between Ascend and NPU?

A: NPU refers to a dedicated processor for neural network algorithms. Different companies have different NPU architectures. Ascend is an NPU processor based on the DaVinci architecture developed by Huawei.

<br/>

## Q: What if I get stuck in model encryption process?

A: First, figure out if the problem is due to the model size it self. As encryption usually costs time, we can estimate the expected time for encryption stage using model size and encryption speed of our machine. If the cost is far more than the expected time, we should consider the problem with secure random number generation by checking the status of system entropy pool. Using Linux as an example, we first query the system entropy threshold by executing `cat /proc/sys/kernel/random/read_wakeup_threshold`. Then we query the number of the currently available entropy using `cat /proc/sys/kernel/random/entropy_avail`. If the number of the currently available entropy is always smaller than system entropy threshold, we can confirm that the problem is caused by the lack of system entropy. In this case, we suggest launch the system entropy gathering and expansion service like `haveged`, in order to accelerate the update speed of system entropy pool.
