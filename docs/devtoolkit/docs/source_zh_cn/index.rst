MindSpore Dev Toolkit文档
============================

MindSpore Dev Toolkit是一款面向MindSpore开发者的开发套件。通过深度学习、智能搜索及智能推荐等技术，打造智能计算最佳体验，致力于全面提升MindSpore框架的易用性，助力MindSpore生态推广。

MindSpore Dev Toolkit目前提供 `创建项目 <https://www.mindspore.cn/devtoolkit/docs/zh-CN/master/mindspore_project_wizard.html>`_ 、`智能补全 <https://www.mindspore.cn/devtoolkit/docs/zh-CN/master/smart_completion.html>`_ 、`API互搜 <https://www.mindspore.cn/devtoolkit/docs/zh-CN/master/operator_search.html>`_ 和 `文档搜索 <https://www.mindspore.cn/devtoolkit/docs/zh-CN/master/knowledge_search.html>`_ 四项功能。

系统需求
------------------------------

MindSpore Dev Toolkit 是一款 `PyCharm <https://www.jetbrains.com/pycharm/>`_ 插件。PyCharm是一款多平台Python IDE。

- 插件支持的操作系统：

  - Windows 10

  - Linux

  - MacOS（仅支持x86架构，补全功能暂未上线）

- 插件支持的PyCharm版本:

  - 2020.3

  - 2021.1

  - 2021.2

  - 2021.3

安装
----------------------------

1. 获取插件安装包

   1.1 下载 `插件Zip包 <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.0/IdePlugin/any/MindSpore_Dev_ToolKit-1.8.0.zip>`_ 。

   1.2 参见下文源码构建章节

2. 启动Pycharm单击左上菜单栏，选择File->Settings->Plugins->Install Plugin from Disk。
   如图：

   .. image:: ./images/clip_image050.jpg

3. 选择下载好的插件包（以1.7版本安装包为例）。

   .. image:: ./images/clip_image052.jpg

4. 插件安装成功。

   .. image:: ./images/clip_image054.jpg

源码构建
----------------------------

请阅读 `源码编译指导 <https://www.mindspore.cn/devtoolkit/docs/zh-CN/master/compiling.html>`_ 。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 使用指南
   :hidden:

   compiling
   mindspore_project_wizard
   operator_search
   knowledge_search
   smart_completion

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
