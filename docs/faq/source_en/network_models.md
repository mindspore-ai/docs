# Network Models

`Data Processing` `Environmental Setup` `Model Export` `Model Training` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_en/network_models.md" target="_blank"><img src="./_static/logo_source.png"></a>

Q: What framework models and formats can be directly read by MindSpore? Can the PTH Model Obtained Through Training in PyTorch Be Loaded to the MindSpore Framework for Use?

A: MindSpore uses protocol buffers (protobuf) to store training parameters and cannot directly read framework models. A model file stores parameters and their values. You can use APIs of other frameworks to read parameters, obtain the key-value pairs of parameters, and load the key-value pairs to MindSpore. If you want to use the .ckpt file trained by a framework, read the parameters and then call the `save_checkpoint` API of MindSpore to save the file as a .ckpt file that can be read by MindSpore.

<br/>

Q: How do I use models trained by MindSpore on Ascend 310? Can they be converted to models used by HiLens Kit?

A: Yes. HiLens Kit uses Ascend 310 as the inference core. Therefore, the two questions are essentially the same. Ascend 310 requires a dedicated OM model. Use MindSpore to export the ONNX or AIR model and convert it into an OM model supported by Ascend 310. For details, see [Multi-platform Inference](https://www.mindspore.cn/tutorial/inference/en/master/multi_platform_inference_ascend_310.html).

<br/>

Q: How do I modify parameters (such as the dropout value) on MindSpore?

A: When building a network, use `if self.training: x = dropput(x)`. During verification, set `network.set_train(mode_false)` before execution to disable the dropout function. During training, set `network.set_train(mode_false)` to True to enable the dropout function.

<br/>

Q: Where can I view the sample code or tutorial of MindSpore training and inference?

A: Please visit the [MindSpore official website training](https://www.mindspore.cn/tutorial/training/en/master/index.html) and [MindSpore official website inference](https://www.mindspore.cn/tutorial/inference/en/master/index.html).

<br/>

Q: What types of model is currently supported by MindSpore for training?

A: MindSpore has basic support for common training scenarios, please refer to [Release note](https://gitee.com/mindspore/mindspore/blob/master/RELEASE.md#) for detailed information.

<br/>

Q: What are the available recommendation or text generation networks or models provided by MindSpore?

A: Currently, recommendation models such as Wide & Deep, DeepFM, and NCF are under development. In the natural language processing (NLP) field, Bert\_NEZHA is available and models such as MASS are under development. You can rebuild the network into a text generation network based on the scenario requirements. Please stay tuned for updates on the [MindSpore Model Zoo](https://gitee.com/mindspore/mindspore/tree/master/model_zoo).

<br/>

Q: How simple can the MindSpore model training code be?

A: MindSpore provides Model APIs except for network definitions. In most scenarios, model training can be completed using only a few lines of code.