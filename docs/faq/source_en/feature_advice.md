# Feature Advice

`Linux` `Windows` `Ascend` `GPU` `CPU` `Environment Preparation` `Basic` `Intermediate`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_en/feature_advice.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

<font size=3>**Q: If MindSpore and PyTorch are installed in an environment, can the syntax of the two frameworks be used together in a Python file?**</font>

A: You can use the two frameworks in a python file. Pay attention to the differences between types. For example, the tensor types created by the two frameworks are different, but the basic types of Python are general.

<br/>

<font size=3>**Q: Can MindSpore read a TensorFlow checkpoint?**</font>

A: The checkpoint format of MindSpore is different from that of TensorFlow. Although both use the Protocol Buffers, their definitions are different. Currently, MindSpore cannot read the TensorFlow or Pytorch checkpoints.

<br/>

<font size=3>**Q: How do I use models trained by MindSpore on Ascend 310? Can they be converted to models used by HiLens Kit?**</font>

A: Yes. HiLens Kit uses Ascend 310 as the inference core. Therefore, the two questions are essentially the same. Ascend 310 requires a dedicated OM model. Use MindSpore to export the ONNX or AIR model and convert it into an OM model supported by Ascend 310. For details, see [Multi-platform Inference](https://www.mindspore.cn/tutorial/inference/en/master/multi_platform_inference_ascend_310.html).

<br/>

<font size=3>**Q: Can MindSpore be converted to an AIR model on Ascend 310?**</font>

A: An AIR model cannot be exported from the Ascend 310. You need to load a trained checkpoint on the Ascend 910, export an AIR model, and then convert the AIR model into an OM model for inference on the Ascend 310. For details about the Ascend 910 installation, see the MindSpore Installation Guide at [here](https://www.mindspore.cn/install/en).

<br/>

<font size=3>**Q: Does MindSpore have any limitation on the input size of a single Tensor for exporting and loading models?**</font>

A: Due to hardware limitations of ProtoBuf, when exporting AIR and ONNX models, the size of a single Tensor cannot exceed 2G. When loading the MindIR model, a single Tensor cannot exceed 2G.

<br/>

<font size=3>**Q: Does MindSpore require computing units such as GPUs and NPUs? What hardware support is required?**</font>

A: MindSpore currently supports CPU, GPU, Ascend, and NPU. Currently, you can try out MindSpore through Docker images on laptops or in environments with GPUs. Some models in MindSpore Model Zoo support GPU-based training and inference, and other models are being improved. For distributed parallel training, MindSpore supports multi-GPU training. You can obtain the latest information from [Road Map](https://www.mindspore.cn/doc/note/en/master/roadmap.html) and [project release notes](https://gitee.com/mindspore/mindspore/blob/master/RELEASE.md#).

<br/>

<font size=3>**Q: Does MindSpore have any plan on supporting other types of heterogeneous computing hardwares?**</font>

A: MindSpore provides pluggable device management interface so that developer could easily integrate other types of heterogeneous computing hardwares like FPGA to MindSpore. We welcome more backend support in MindSpore from the community.

<br/>

<font size=3>**Q: The recent announced programming language such as taichi got Python extensions that could be directly used as `import taichi as ti`. Does MindSpore have similar support?**</font>

A: MindSpore supports Python native expression via `import mindspore`.

<br/>

<font size=3>**Q: Does MindSpore support truncated gradient?**</font>

A: Yes. For details, see [Definition and Usage of Truncated Gradient](https://gitee.com/mindspore/mindspore/blob/master/model_zoo/official/nlp/transformer/src/transformer_for_train.py#L35).

<br/>

<font size=3>**Q: What is the MindSpore IR design concept?**</font>

A: Function expression: All expressions are functions, and differentiation and automatic parallel analysis are easy to implement without side effect. `JIT` compilation capability: The graph-based IR, control flow dependency, and data flow are combined to balance the universality and usability. Turing-complete IR: More flexible syntaxes are provided for converting `Python`, such as recursion.

<br/>

<font size=3>**Q: What are the advantages and features of MindSpore parallel model training?**</font>

A: In addition to data parallelism, MindSpore distributed training also supports operator-level model parallelism. The operator input tensor can be tiled and parallelized. On this basis, automatic parallelism is supported. You only need to write a single-device script to automatically tile the script to multiple nodes for parallel execution.

<br/>

<font size=3>**Q: How does MindSpore implement semantic collaboration and processing? Is the popular Formal Concept Analysis (FCA) used?**</font>

A: The MindSpore framework does not support FCA. For semantic models, you can call third-party tools to perform FCA in the data preprocessing phase. MindSpore supports Python therefore `import FCA` could do the trick.

<br/>

<font size=3>**Q: Does MindSpore have any plan or consideration on the edge and device when the training and inference functions on the cloud are relatively mature?**</font>

A: MindSpore is a unified cloud-edge-device training and inference framework. Edge has been considered in its design, so MindSpore can perform inference at the edge. The open-source version will support Ascend 310-based inference. The optimizations supported in the current inference stage include quantization, operator fusion, and memory overcommitment.

<br/>

<font size=3>**Q: How does MindSpore support automatic parallelism?**</font>

A: Automatic parallelism on CPUs and GPUs are being improved. You are advised to use the automatic parallelism feature on the Ascend 910 AI processor. Follow our open source community and apply for a MindSpore developer experience environment for trial use.

<br/>

<font size=3>**Q: Does MindSpore have a module that can implement object detection algorithms as TensorFlow does?**</font>

A: The TensorFlow's object detection pipeline API belongs to the TensorFlow's Model module. After MindSpore's detection models are complete, similar pipeline APIs will be provided.

<br/>

<font size=3>**Q: Does MindSpore Serving support hot loading to avoid inference service interruption?**</font>

A: MindSpore does not support hot loading. It is recommended that you run multiple Serving services and restart some of them when switching the version.

<br/>

<font size=3>**Q: Does MindSpore Serving allow multiple workers to be started for one model to support multi-device and single-model concurrency?**</font>

A: MindSpore Serving does not support distribution and this function is being developed. That is, multiple workers cannot be started for one model. It is recommended that multiple Serving services be deployed to implement distribution and load balancing. In addition, to avoid message forwarding between `master` and `worker`, you can use the `start_servable_in_master` API to enable `master` and `worker` to be executed in the same process, implementing lightweight deployment of the Serving services.

<br/>

<font size=3>**Q: How does the MindSpore Serving version match the MindSpore version?**</font>

A: MindSpore Serving matches MindSpore in the same version. For example, Serving `1.1.1` matches MindSpore `1.1.1`.

<br/>

<font size=3>**Q: How do I perform transfer learning in PyNative mode?**</font>

A: PyNative mode is compatible with transfer learning. For more tutorial information, see [Code for Loading a Pre-Trained Model](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/cv_mobilenetv2_fine_tune.html#code-for-loading-a-pre-trained-model).

<br/>

<font size=3>**Q:What is the difference between [MindSpore ModelZoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo) and [Ascend ModelZoo](https://www.hiascend.com/software/modelzoo)?**</font>

A: `MindSpore ModelZoo` contains models only implemented by MindSpore. But these models support different devices including Ascend, GPU, CPU and mobile. `Ascend ModelZoo` contains models only running on Ascend which are implemented by different ML platform including MindSpore, PyTorch, TensorFlow and Caffe. You can refer to the corresponding [gitee repository](https://gitee.com/ascend/modelzoo).

As for the models implemented by MindSpore running on Ascend, these are maintained in `MindSpore ModelZoo`, and will be released to `Ascend ModelZoo` regularly.
