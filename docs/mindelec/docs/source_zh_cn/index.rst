MindElec介绍
==============

电磁仿真是指通过计算的方式模拟电磁波在物体或空间中的传播特性，其在手机容差、天线优化和芯片设计等场景中应用广泛。传统数值方法如有限差分、有限元等需网格剖分、迭代计算，仿真流程复杂、计算时间长，无法满足产品的设计需求。AI方法具有万能逼近和高效推理能力，可有效提升仿真效率。

MindElec是基于MindSpore开发的AI电磁仿真工具包，由数据构建及转换、仿真计算、以及结果可视化组成。可以支持端到端的AI电磁仿真。目前已在华为终端手机容差场景中取得阶段性成果，相比商业仿真软件，AI电磁仿真的S参数误差在2%左右，端到端仿真速度提升10+倍。

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindelec/docs/source_zh_cn/images/MindElec-architecture.jpg" width="600px" alt="" >

数据构建及转换
----------------

支持CSG （Constructive Solid Geometry，CSG）
模式的几何构建，如矩形、圆形等结构的交集、并集和差集，以及cst和stp数据（CST等商业软件支持的数据格式）的高效张量转换。未来还会支持智能网格剖分，为传统科学计算使用。

仿真计算
----------

AI电磁模型库
^^^^^^^^^^^^^

提供物理和数据驱动的AI电磁模型：物理驱动是指网络的训练无需额外的标签数据，只需方程和初边界条件即可；数据驱动是指训练需使用仿真或实验等产生的数据。物理驱动相比数据驱动，优势在于可避免数据生成带来的成本和网格独立性等问题，劣势在于需明确方程的具体表达形式并克服点源奇异性、多任务损失函数以及泛化性等技术挑战。

优化策略
^^^^^^^^

为提升物理和数据驱动模型的精度、减少训练的成本，提供了一系列优化措施。数据压缩可以有效地减少神经网络的存储和计算量；多尺度滤波、动态自适应加权可以提升模型的精度，克服点源奇异性等问题；小样本学习主要是为了减少训练的数据量，节省训练的成本。

结果可视化
----------

仿真的结果如S参数或电磁场等可保存在CSV、VTK文件中。MindInsight可以显示训练过程中的损失函数变化，并以图片的形式在网页上展示结果；Paraview是第三方开源软件，具有动态展示切片、翻转等高级功能。

.. toctree::
   :maxdepth: 1
   :caption: 安装部署

   intro_and_install

.. toctree::
   :maxdepth: 1
   :caption: 应用实践

   physics_driven
   data_driven
   AD_FDTD
   visualization

.. toctree::
   :maxdepth: 1
   :caption: API参考

   mindelec.architecture
   mindelec.common
   mindelec.data
   mindelec.geometry
   mindelec.loss
   mindelec.operators
   mindelec.solver
   mindelec.vision
