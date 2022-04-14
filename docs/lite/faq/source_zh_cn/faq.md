# FAQ

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/lite/faq/source_zh_cn/faq.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.7/resource/_static/logo_source.png"></a>

<br/>

<font size=3>**Q：MindSpore Lite支持的日志级别有几种？怎么设置日志级别？**</font>

A：目前支持DEBUG、INFO、WARNING、ERROR四种日志级别，用户可以通过设置环境变量GLOG_v为0~3选择打印的日志级别，0~3分别对应DEBUG、INFO、WARNING和ERROR，默认打印WARNING和ERROR级别的日志。例如设置GLOG_v为1即可打印INFO及以上级别的日志。

<br/>

<font size=3>**Q：NPU推理存在什么限制？**</font>

A：目前NPU仅支持在系统ROM版本EMUI>=11、芯片支持包括Kirin 9000、Kirin 9000E、Kirin 990、Kirin 985、Kirin 820、Kirin 810等，具体约束和芯片支持请查看：<https://developer.huawei.com/consumer/cn/doc/development/hiai-Guides/mapping-relationship-0000001052830507#ZH-CN_TOPIC_0000001052830507__section94427279718>

<br/>

<font size=3>**Q：为什么使用裁剪工具裁剪后的静态库在集成时存在编译失败情况？**</font>

A：目前裁剪工具仅支持CPU和GPU的库，具体使用请查看[使用裁剪工具降低库文件大小](https://www.mindspore.cn/lite/docs/zh-CN/r1.7/use/cropper_tool.html)文档。

<br/>

<font size=3>**Q：MindSpore Lite推理是否会耗尽手机全部内存?**</font>

A：MindSpore Lite内置内存池有最大容量限制，为3GB，如果模型较大，超过最大容量限制，运行将会异常退出。

<br/>

<font size=3>**Q：MindSpore Lite的离线模型MS文件如何进行可视化，看到网络结构？**</font>

A：模型可视化开源仓库`Netron`已经支持查看MindSpore Lite模型（MindSpore版本 >= r1.2），请到Netron官网下载安装包[Netron](https://github.com/lutzroeder/netron)。

<br/>

<font size=3>**Q：MindSpore有量化推理工具么？**</font>

A：[MindSpore Lite](https://www.mindspore.cn/lite)支持云侧量化感知训练的量化模型的推理，MindSpore Lite converter工具提供训练后量化以及权重量化功能，且功能在持续加强完善中。

<br/>

<font size=3>**Q：MindSpore有轻量的端侧推理引擎么？**</font>

A：MindSpore轻量化推理框架MindSpore Lite已于r0.7版本正式上线，欢迎试用并提出宝贵意见，概述、教程和文档等请参考[MindSpore Lite](https://www.mindspore.cn/lite)

<br/>

<font size=3>**Q：针对编译JAVA库时出现 `sun.security.validator.ValidatorException: PKIX path building failed: sun.security.provider.certpath.SunCertPathBuilderException: unable to find valid certification path to requested target` 问题时如何解决？**</font>

A：需要使用keytool工具将相关网站的安全证书导入java的cacerts证书库 `keytool -import -file "XX.cer" -keystore ${JAVA_HOME}/lib/security/cacerts" -storepass changeit`。

<br/>