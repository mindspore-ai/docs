# Supported Features

`Characteristic Advantages` `On-device Inference` `Functional Module` `Reasoning Tools`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_en/supported_features.md" target="_blank"><img src="./_static/logo_source.png"></a>

Q: Does MindSpore have a quantized inference tool?

A: [MindSpore Lite](https://www.mindspore.cn/lite/en) supports the inference of the quantization aware training model on the cloud. The MindSpore Lite converter tool provides the quantization after training and weight quantization functions which are being continuously improved.

<br/>

Q: What are the advantages and features of MindSpore parallel model training?

A: In addition to data parallelism, MindSpore distributed training also supports operator-level model parallelism. The operator input tensor can be tiled and parallelized. On this basis, automatic parallelism is supported. You only need to write a single-device script to automatically tile the script to multiple nodes for parallel execution.

<br/>

Q: Has MindSpore implemented the anti-pooling operation similar to `nn.MaxUnpool2d`?
A: Currently, MindSpore does not provide anti-pooling APIs but you can customize the operator to implement the operation. For details, click [here](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/custom_operator_ascend.html).

<br/>

Q: Does MindSpore have a lightweight on-device inference engine?

A:The MindSpore lightweight inference framework MindSpore Lite has been officially launched in r0.7. You are welcome to try it and give your comments. For details about the overview, tutorials, and documents, see [MindSpore Lite](https://www.mindspore.cn/lite/en).

<br/>

Q: How does MindSpore implement semantic collaboration and processing? Is the popular Formal Concept Analysis (FCA) used?

A: The MindSpore framework does not support FCA. For semantic models, you can call third-party tools to perform FCA in the data preprocessing phase. MindSpore supports Python therefore `import FCA` could do the trick.

<br/>

Q: Does MindSpore have any plan or consideration on the edge and device when the training and inference functions on the cloud are relatively mature?

A: MindSpore is a unified cloud-edge-device training and inference framework. Edge has been considered in its design, so MindSpore can perform inference at the edge. The open-source version will support Ascend 310-based inference. The optimizations supported in the current inference stage include quantization, operator fusion, and memory overcommitment.

<br/>

Q: How does MindSpore support automatic parallelism?

A: Automatic parallelism on CPUs and GPUs are being improved. You are advised to use the automatic parallelism feature on the Ascend 910 AI processor. Follow our open source community and apply for a MindSpore developer experience environment for trial use.

<br/>

Q: Does MindSpore have a module that can implement object detection algorithms as TensorFlow does?

A: The TensorFlow's object detection pipeline API belongs to the TensorFlow's Model module. After MindSpore's detection models are complete, similar pipeline APIs will be provided.

<br/>

Q: How do I migrate scripts or models of other frameworks to MindSpore?

A: For details about script or model migration, please visit the [MindSpore official website](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/migrate_3rd_scripts.html).

<br/>

Q: Does MindSpore provide open-source e-commerce datasets?

A: No. Please stay tuned for updates on the [MindSpore official website](https://www.mindspore.cn/en).
