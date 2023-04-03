MindSpore Dev Toolkit文档
============================

MindSpore Dev Toolkit是一款支持MindSpore开发的 `PyCharm <https://www.jetbrains.com/pycharm/>`_ （多平台Python IDE）插件，提供 `创建项目 <https://www.mindspore.cn/devtoolkit/docs/zh-CN/r2.0/mindspore_project_wizard.html>`_ 、 `智能补全 <https://www.mindspore.cn/devtoolkit/docs/zh-CN/r2.0/smart_completion.html>`_ 、 `API互搜 <https://www.mindspore.cn/devtoolkit/docs/zh-CN/r2.0/operator_search.html>`_ 和 `文档搜索 <https://www.mindspore.cn/devtoolkit/docs/zh-CN/r2.0/knowledge_search.html>`_ 等功能。

MindSpore Dev Toolkit通过深度学习、智能搜索及智能推荐等技术，打造智能计算最佳体验，致力于全面提升MindSpore框架的易用性，助力MindSpore生态推广。

系统需求
------------------------------

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

   1.1 下载 `插件Zip包 <https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.0.0/IdePlugin/any/MindSpore_Dev_ToolKit-2.0.0.zip>`_ 。

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

请阅读 `源码编译指导 <https://www.mindspore.cn/devtoolkit/docs/zh-CN/r2.0/compiling.html>`_ 。

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: PyCharm插件使用指南
   :hidden:

   PyCharm_plugin_install
   compiling
   smart_completion
   operator_search
   operator_scanning
   knowledge_search
   mindspore_project_wizard

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: VSCode插件使用指南
   :hidden:

   VSCode_plugin_install
   VSCode_smart_completion

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE