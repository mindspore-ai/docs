# Feature Advice

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/faq/feature_advice.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: Is the `input=np.random.uniform(...)` format fixed when the MindIR format is exported?**</font>

A: The format is not fixed. This step is to create an input for constructing the network structure. You only need to input the correct `shape` in `export`. You can use `np.ones` and `np.zeros` to create an input.

<br/>

<font size=3>**Q: What framework models and formats can be directly read by MindSpore? Can the PTH Model obtained through training in PyTorch be loaded to the MindSpore framework for use?**</font>

A: MindSpore uses Protobuf to store training parameters and cannot directly read framework models. A model file stores parameters and their values. You can use APIs of other frameworks to read parameters, obtain the key-value pairs of parameters, and load the key-value pairs to MindSpore. If you want to use the .ckpt file trained by other framework, read the parameters and then call the `save_checkpoint` API of MindSpore to save the file as a .ckpt file that can be read by MindSpore.

<br/>

<font size=3>**Q: What should I do if a Protobuf memory limit error is reported during the process of using ckpt or exporting a model?**</font>

A: When a single Protobuf data is too large, because Protobuf itself limits the size of the data stream, a memory limit error will be reported. At this time, the restriction can be lifted by setting the environment variable `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`.

<br/>

<font size=3>**Q: What is the difference between the PyNative and Graph modes?**</font>

A: Compare through the following four aspects:

- In terms of network execution: operators used in the two modes are the same. Therefore, when the same network and operators are executed in the two modes, the accuracy is the same. As Graph mode uses graph optimization, calculation graph sinking and other technologies, it has higher performance and efficiency in executing the network.

- In terms of application scenarios: Graph mode requires the network structure to be built at the beginning, and then the framework performs entire graph optimization and execution. This mode is suitable to scenarios where the network is fixed and high performance is required.

- In term of different hardware (such as `Ascend`, `GPU`, and `CPU`) resources: the two modes are supported.

- In terms of code debugging: since operators are executed line by line in PyNative mode, you can directly debug the Python code and view the `/api` output or execution result of the corresponding operator at any breakpoint in the code. In Graph mode, the network is built but not executed in the constructor function. Therefore, you cannot obtain the output of the corresponding operator at breakpoints in the `construct` function. You can only specify operators and print their output results, and then view the results after the network execution is completed.

<br/>

<font size=3>**Q: Does MindSpore run only on Huawei `Ascend`?**</font>

A: MindSpore supports Huawei `Ascend`, `GPUs`, and `CPUs`, and supports heterogeneous computing.

<br/>

<font size=3>**Q: If MindSpore and PyTorch are installed in an environment, can the syntax of the two frameworks be used together in a Python file?**</font>

A: You can use the two frameworks in a python file. Pay attention to the differences between types. For example, the tensor types created by the two frameworks are different, but the basic types of Python are general.

<br/>

<font size=3>**Q: Can MindSpore read a ckpt file of TensorFlow?**</font>

A: The formats of  `ckpt` of MindSpore and `ckpt`of TensorFlow are not generic. Although both use the `Protocol` Buffers, the definition of `proto` are different. Currently, MindSpore cannot read the TensorFlow or Pytorch `ckpt` files.

<br/>

<font size=3>**Q: How do I use models trained by MindSpore on Ascend 310? Can they be converted to models used by HiLens Kit?**</font>

A: Yes. HiLens Kit uses Ascend 310 as the inference core. Therefore, the two questions are essentially the same, which both need to convert as OM model. Ascend 310 requires a dedicated OM model. Use MindSpore to export the ONNX or AIR model and convert it into an OM model supported by Ascend 310. For details, see [Multi-platform Inference](https://www.mindspore.cn/tutorials/experts/en/r1.7/infer/inference.html).

<br/>

<font size=3>**Q: Does MindSpore only be run on Huawei own `Ascend`?**</font>

A: MindSpore supports Huawei's own `Ascend`, `GPU` and `CPU` at the same time, and supports heterogeneous computing power.

<br/>

<font size=3>**Q: Can MindSpore be converted to an AIR model on Ascend 310?**</font>

A: An AIR cannot be exported from the Ascend 310. You need to load a trained checkpoint on the Ascend 910, export an AIR model, and then convert the AIR model into an OM model for inference on the Ascend 310. For details about the Ascend 910 installation, see the MindSpore Installation Guide at [here](https://www.mindspore.cn/install/en).

<br/>

<font size=3>**Q: Does MindSpore have any limitation on the input size of a single Tensor for exporting and loading models?**</font>

A: Due to hardware limitations of Protobuf, when exporting to AIR and ONNX formats, the size of model parameters cannot exceed 2G; when exporting to MINDIR format, there is no limit to the size of model parameters. MindSpore only supports MINDIR and doesn't support AIR and ONNX formats. The import size limitation is the same as that of export.

<br/>

<font size=3>**Q: Does MindSpore need a GPU computing unit? What hardware support is needed?**</font>

A: MindSpore currently supports CPU, GPU, and Ascend. Currently, you can try out MindSpore through Docker images on laptops or in environments with GPUs. Some models in MindSpore Model Zoo support GPU-based training and inference, and other models are being improved. For distributed parallel training, MindSpore supports multi-GPU training. You can obtain the latest information from [project release notes](https://gitee.com/mindspore/mindspore/blob/r1.7/RELEASE.md#).

<br/>

<font size=3>**Q: Does MindSpore have any plan on supporting heterogeneous computing hardwares?**</font>

A: MindSpore provides pluggable device management interface, so that developer could easily integrate other types of heterogeneous computing hardwares (like FPGA) to MindSpore. We welcome more backend support in MindSpore from the community.

<br/>

<font size=3>**Q: What is the relationship between MindSpore and ModelArts? Can we use MindSpore in ModelArts?**</font>

A: ModelArts is Huawei public cloud online training and inference platform, and MindSpore is Huawei deep learning framework. The tutorial shows how users can use ModelArts to train ModelsSpore models in detail.

<br/>

<font size=3>**Q: The recent announced programming language such as taichi got Python extensions that could be directly used as `import taichi as ti`. Does MindSpore have similar support?**</font>

A: MindSpore supports Python native expression and `import mindspore` related package can be used.

<br/>

<font size=3>**Q: Does MindSpore support truncated gradient?**</font>

A: Yes. For details, see [Definition and Usage of Truncated Gradient](https://gitee.com/mindspore/models/blob/r1.7/official/nlp/transformer/src/transformer_for_train.py#L35).

<br/>

<font size=3>**Q: What is the MindSpore IR design concept?**</font>

A: Function expression: All expressions are functions, and differentiation and automatic parallel analysis are easy to implement without side effect. `JIT` compilation capability: The graph-based IR, control flow dependency, and data flow are combined to balance the universality and usability. Graphically complete IR: More conversion `Python` flexible syntax, including recursion, etc.

<br/>

<font size=3>**Q: What are the advantages and features of MindSpore parallel model training?**</font>

A: In addition to data parallelism, MindSpore distributed training also supports operator-level model parallelism. The operator input tensor can be tiled and parallelized. On this basis, automatic parallelism is supported. You only need to write a single-device script to automatically tile the script to multiple nodes for parallel execution.

<br/>

<font size=3>**Q: How does MindSpore implement semantic collaboration and processing? Is the popular Formal Concept Analysis (FCA) used?**</font>

A: The MindSpore framework does not support FCA. For semantic models, you can call third-party tools to perform FCA in the data preprocessing phase. MindSpore supports Python therefore `import FCA` related package could do the trick.

<br/>

<font size=3>**Q: Does MindSpore have any plan on the edge and device when the training and inference functions of MindSpore on the cloud are relatively mature?**</font>

A: MindSpore is a unified cloud-edge-device training and inference framework, which supports exporting cloud-side trained models to Ascend AI processors and terminal devices for inference. The optimizations supported in the current inference stage include quantization, operator fusion, and memory overcommitment.

<br/>

<font size=3>**Q: How does MindSpore support automatic parallelism?**</font>

A: Automatic parallelism on CPUs and GPUs are being improved. You are advised to use the automatic parallelism on the Ascend 910 AI processor. Follow our open source community and apply for a MindSpore developer experience environment for trial use.

<br/>

<font size=3>**Q: Does MindSpore have a similar module that can implement object detection algorithms based on TensorFlow?**</font>

A: The TensorFlow's object detection Pipeline API belongs to the TensorFlow's Model module. After MindSpore's detection models are complete, similar Pipeline APIs will be provided.

<br/>

<font size=3>**Q: How do I perform transfer learning in PyNative mode?**</font>

A: PyNative mode is compatible with transfer learning.

<br/>

<font size=3>**Q: What is the difference between [MindSpore ModelZoo](https://gitee.com/mindspore/models/tree/master) and [Ascend ModelZoo](https://www.hiascend.com/software/modelzoo)?**</font>

A: `MindSpore ModelZoo` contains models mainly implemented by MindSpore. But these models support different devices including Ascend, GPU, CPU and Mobile. `Ascend ModelZoo` contains models only running on Ascend which are implemented by different ML platform including MindSpore, PyTorch, TensorFlow and Caffe. You can refer to the corresponding [gitee repository](https://gitee.com/ascend/modelzoo).

The combination of MindSpore and Ascend is overlapping, and this part of the model will be based on MindSpore's ModelZoo as the main version, and will be released to Ascend ModelZoo regularly.

<br/>

<font size=3>**Q: What is the relationship between Ascend and NPU?**</font>

A: NPU refers to a dedicated processor for neural network algorithms. Different companies have different NPU architectures. Ascend is an NPU processor based on the DaVinci architecture developed by Huawei.
