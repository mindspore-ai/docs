香橙派开发
===============

`OrangePi AIpro(香橙派 AIpro) <http://www.orangepi.cn/index.html>`_ 采用昇腾AI技术路线，具体为4核64位处理器和AI处理器，集成图形处理器。

目前已实现OrangePi AIpro开发板的系统镜像预装昇思MindSpore AI框架，并在后续版本迭代中持续演进，当前已完整支持昇思MindSpore官网教程中的所有网络模型。OrangePi AIpro开发板向开发者提供的官方系统镜像有openEuler版本和Ubuntu版本，两个镜像版本均已预置昇思MindSpore，便于用户体验软硬协同优化后带来的高效开发体验。同时，欢迎开发者自定义配置昇思MindSpore和CANN运行环境。

香橙派开发前，需要了解以下内容：

- `昇思MindSpore <https://www.mindspore.cn/>`_ 
- `Linux <https://www.runoob.com/linux/linux-tutorial.html>`_ 
- `Jupyter Notebook <https://jupyter.org/documentation>`_ 

接下来的教程将演示如何基于OrangePi AIpro进行自定义环境搭建，如何在OrangePi AIpro启动Jupyter Lab，并以手写数字识别为例，介绍OrangePi AIpro上基于昇思MindSpore进行在线推理需要完成的操作。

以下操作基于OrangePi AIpro 8-12TOPS 16G开发板，20TOPS开发板操作方式相同。

更多基于昇思MindSpore的香橙派开发板案例详见： `GitHub链接 <https://github.com/mindspore-courses/orange-pi-mindspore>`_ 
