MindSpore Flow介绍
==================

流体仿真是指通过数值计算对给定边界条件下的流体控制方程进行求解，从而实现流动的分析、预测和控制，其在航空航天、船舶制造以及能源电力等行业领域的工程设计中应用广泛。传统流体仿真的数值方法如有限体积、有限差分等，主要依赖商业软件实现，需要进行物理建模、网格划分、数值离散、迭代求解等步骤，仿真过程较为复杂，计算周期长。AI具备强大的学习拟合和天然的并行推理能力，可以有效地提升流体仿真效率。

`MindSpore Flow <https://gitee.com/mindspore/mindscience/tree/master/MindFlow>`_ 是基于昇思MindSpore开发的流体仿真领域套件，支持航空航天、船舶制造以及能源电力等行业领域的AI流场模拟，旨在于为广大的工业界科研工程人员、高校老师及学生提供高效易用的AI计算流体仿真软件。

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindflow/docs/source_zh_cn/images/mindflow_archi_cn.png" width="1200px" alt="" style="display: inline-block">

代码仓地址: <https://gitee.com/mindspore/mindscience/tree/master/MindFlow>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   mindflow_install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 特性

   features/solve_pinns_by_mindflow

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 物理驱动

   physics_driven/burgers1D
   physics_driven/darcy2D
   physics_driven/navier_stokes2D
   physics_driven/poisson_geometry
   physics_driven/taylor_green2D
   physics_driven/navier_stokes_inverse
   physics_driven/boltzmann
   physics_driven/kovasznay
   physics_driven/periodic_hill
   physics_driven/poisson_point_source

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 数据驱动

   data_driven/2D_steady
   data_driven/burgers_FNO1D
   data_driven/navier_stokes_FNO2D
   data_driven/burgers_KNO1D
   data_driven/navier_stokes_KNO2D
   data_driven/2D_unsteady
   data_driven/flow_around_sphere
   data_driven/navier_stokes_FNO3D
   data_driven/burgers_SNO1D
   data_driven/navier_stokes_SNO2D
   data_driven/navier_stokes_SNO3D

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 数据机理融合

   data_mechanism_fusion/pde_net
   data_mechanism_fusion/percnn2d
   data_mechanism_fusion/percnn3d
   data_mechanism_fusion/phympgn

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 可微分CFD求解器

   cfd_solver/lax_tube
   cfd_solver/sod_tube
   cfd_solver/couette
   cfd_solver/riemann2d
   cfd_solver/acoustic

.. toctree::
   :maxdepth: 1
   :caption: API参考

   mindflow.cell
   mindflow.cfd
   mindflow.core
   mindflow.data
   mindflow.geometry
   mindflow.pde

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
