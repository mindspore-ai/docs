MindSpore Dev Toolkit
============================

MindSpore Dev Toolkit is a development kit for MindSpore developers. It uses technologies such as deep learning, intelligent search, and intelligent recommendation to build the best intelligent computing experience, improve the usability of the MindSpore framework, and facilitate the promotion of the MindSpore ecosystem.

MindSpore Dev Toolkit currently provides `Project creation <https://www.mindspore.cn/devtoolkit/docs/en/r1.8/mindspore_project_wizard.html>`_, `Code completion <https://www.mindspore.cn/devtoolkit/docs/en/r1.8/smart_completion.html>`_, `Operator search <https://www.mindspore.cn/devtoolkit/docs/en/r1.8/operator_search.html>`_, `Document search <https://www.mindspore.cn/devtoolkit/docs/en/r1.8/knowledge_search.html>`_.

System Requirements
------------------------------

MindSpore Dev Toolkit is a `PyCharm <https://www.jetbrains.com/pycharm/>`_ plug-in. PyCharm is a cross-platform Python IDE.

- Operating systems supported by the plug-in:

  - Windows 10

  - Linux

  - macOS (Only the x86 architecture is supported. The code completion function is not available currently.)

- PyCharm versions supported by the plug-in:

  - 2020.3

  - 2021.1

  - 2021.2

  - 2021.3

Installation
----------------------------

1. Obtain the plug-in installation package.

   1.1 Download the `plug-in ZIP package <https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.8.0/IdePlugin/any/MindSpore_Dev_ToolKit-1.8.0.zip>`_.

   1.2 See the following section "Source Code Build."

2. Start the PyCharm. On the menu bar in the upper left corner, choose **File** > **Settings** > **Plugins** > **Install Plugin from Disk**.
   See the following figure:

   .. image:: ./images/clip_image050.jpg

3. Select the ZIP package of the plug-in (take version 1.7 as an example).

   .. image:: ./images/clip_image052.jpg

4. The plug-in is installed successfully.

   .. image:: ./images/clip_image054.jpg

Source Code Build
----------------------------

See the `Source Code Compilation Guide <https://www.mindspore.cn/devtoolkit/docs/en/r1.8/compiling.html>`_.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: operating guide
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
