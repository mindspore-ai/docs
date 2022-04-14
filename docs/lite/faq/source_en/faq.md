# FAQ

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/lite/faq/source_en/faq.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source_en.png"></a>

<br/>

<font size=3>**Q：How many log levels are supported by MindSpore Lite? How can I set the log level?**</font>

A：Currently MindSpore Lite supports 4 log levels, including DEBUG, INFO, WARNING and ERROR. Users can set log level by set environment parameter GLOG_v. This environment parameter ranges from 0 to 3, which represents DEBUG, INFO, WARNING and ERROR. The default log level is WARNING or ERROR. For example, if the user sets GLOG_v to 1, MindSpore Lite will print the log of INFO level or higher.
<br/>

<font size=3>**Q： What are the limitations of NPU?**</font>

A： Currently NPU only supports system ROM version EMUI>=11. Chip support includes Kirin 9000, Kirin 9000E, Kirin 990, Kirin 985, Kirin 820, Kirin 810, etc. For specific constraints and chip support, please see: <https://developer.huawei.com/consumer/en/doc/development/hiai-Guides/mapping-relationship-0000001052830507#EN-US_TOPIC_0000001052830507__section94427279718>

<br/>

<font size=3>**Q： Why does the static library after cutting with the cropper tool fail to compile during integration?**</font>

A： Currently the cropper tool only supports CPU and GPU libraries. For details, please refer to [Use clipping tool to reduce library file size](https://www.mindspore.cn/lite/docs/en/r1.7/use/cropper_tool.html) document.

<br/>

<font size=3>**Q： Will MindSpore Lite run out of device memory, when running model?**</font>

A： Currently the MindSpore Lite built-in memory pool has a maximum capacity limit 3GB. If a model is bigger than 3GB, MindSpore Lite will throw error.

<font size=3>**Q: How do I visualize the MindSpore Lite offline model (.ms file) to view the network structure?**</font>

A: Model visualization open-source repository `Netron` supports viewing MindSpore Lite models (MindSpore >= r1.2), which can be downloaded in the [Netron](https://github.com/lutzroeder/netron).

<br/>

<font size=3>**Q: Does MindSpore have a quantized inference tool?**</font>

A: [MindSpore Lite](https://www.mindspore.cn/lite/en) supports the inference of the quantization aware training model on the cloud. The MindSpore Lite converter tool provides the quantization after training and weight quantization functions which are being continuously improved.

<br/>

<font size=3>**Q: Does MindSpore have a lightweight on-device inference engine?**</font>

A:The MindSpore lightweight inference framework MindSpore Lite has been officially launched in r0.7. You are welcome to try it and give your comments. For details about the overview, tutorials, and documents, see [MindSpore Lite](https://www.mindspore.cn/lite/en).

<br/>

<font size=3>**Q: How to solve the problem of `sun.security.validator.ValidatorException: PKIX path building failed: sun.security.provider.certpath.SunCertPathBuilderException: unable to find valid certification path to requested target` when compiling JAVA library?**</font>

A：Use the `keytool` tool to import the security certificate of the relevant website into the cacerts certificate library of java. E.g `keytool -import -file "XX.cer" -keystore ${JAVA_HOME}/lib/security/cacerts" -storepass changeit`.

<br/>