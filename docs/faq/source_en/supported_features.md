# Supported Features

`Characteristic Advantages` `On-device Inference` `Functional Module` `Reasoning Tools`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/tree/r1.0/docs/faq/source_en/supported_features.md)

<font size=3>**Q: How do I change hyperparameters for calculating loss values during neural network training?**</font>

A: Sorry, this function is not available yet. You can find the optimal hyperparameters by training, redefining an optimizer, and then training.

<br/>

<font size=3>**Q: Can you introduce the dedicated data processing framework?**</font>

A: MindData provides the heterogeneous hardware acceleration function for data processing. The high-concurrency data processing pipeline supports NPUs, GPUs, and CPUs. The CPU usage is reduced by 30%. For details, see [Optimizing Data Processing](https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/optimize_data_processing.html).

<br/>

<font size=3>**Q: What is the MindSpore IR design concept?**</font>

A: Function expression: All expressions are functions, and differentiation and automatic parallel analysis are easy to implement without side effect. `JIT compilation capability`: The graph-based IR, control flow dependency, and data flow are combined to balance the universality and usability. `Turing-complete IR`: More flexible syntaxes are provided for converting `Python`, such as recursion.

<br/>

<font size=3>**Q: Will MindSpore provide a reinforcement learning framework?**</font>

A: This function is at the design stage. You can contribute ideas and scenarios and participate in the construction. Thank you.

<br/>

<font size=3>**Q: As Google Colab and Baidu AI Studio provide free `GPU` computing power, does MindSpore provide any free computing power?**</font>

A: If you cooperate with MindSpore in papers and scientific research, you can obtain free cloud computing power. If you want to simply try it out, we can also provide online experience similar to that of Colab.

<br/>

<font size=3>**Q: How do I visualize the MindSpore Lite offline model (.ms file) to view the network structure?**</font>

A: MindSpore Lite code is being submitted to the open-source repository Netron. Later, the MS model visualization will be implemented using Netron. While there are still some issues to be resolved in the Netron open-source repository, we have a Netron version for internal use, which can be [downloaded](https://github.com/lutzroeder/netron/releases).

<br/>

<font size=3>**Q: Does MindSpore have a quantized inference tool?**</font>

A: [MindSpore Lite](https://www.mindspore.cn/lite/en) supports the inference of the quantization aware training model on the cloud. The MindSpore Lite converter tool provides the quantization after training and weight quantization functions which are being continuously improved.

<br/>

<font size=3>**Q: What are the advantages and features of MindSpore parallel model training?**</font>

A: In addition to data parallelism, MindSpore distributed training also supports operator-level model parallelism. The operator input tensor can be tiled and parallelized. On this basis, automatic parallelism is supported. You only need to write a single-device script to automatically tile the script to multiple nodes for parallel execution.

<br/>

<font size=3>**Q: Has MindSpore implemented the anti-pooling operation similar to `nn.MaxUnpool2d`?**</font>

A: Currently, MindSpore does not provide anti-pooling APIs but you can customize the operator to implement the operation. For details, click [here](https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/custom_operator_ascend.html).

<br/>

<font size=3>**Q: Does MindSpore have a lightweight on-device inference engine?**</font>

A:The MindSpore lightweight inference framework MindSpore Lite has been officially launched in r0.7. You are welcome to try it and give your comments. For details about the overview, tutorials, and documents, see [MindSpore Lite](https://www.mindspore.cn/lite/en).

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

<font size=3>**Q: How do I migrate scripts or models of other frameworks to MindSpore?**</font>

A: For details about script or model migration, please visit the [MindSpore official website](https://www.mindspore.cn/tutorial/training/en/r1.0/advanced_use/migrate_3rd_scripts.html).

<br/>

<font size=3>**Q: Does MindSpore provide open-source e-commerce datasets?**</font>

A: No. Please stay tuned for updates on the [MindSpore official website](https://www.mindspore.cn/en).
