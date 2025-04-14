MindSpore Flow Introduction
=============================

Flow simulation aims to solve the fluid governing equation under a given boundary condition by numerical methods, so as to realize the flow analysis, prediction and control. It is widely used in engineering design in aerospace, ship manufacturing, energy and power industries. The numerical methods of traditional flow simulation, such as finite volume method and finite difference method, are mainly implemented by commercial software, requiring physical modeling, mesh generation, numerical dispersion, iterative solution and other steps. The simulation process is complex and the calculation cycle is long. AI has powerful learning fitting and natural parallel inference capabilities, which can improve the efficiency of the flow simulation.

`MindSpore Flow <https://gitee.com/mindspore/mindscience/tree/master/MindFlow>`_ is a flow simulation suite developed based on MindSpore. It supports AI flow simulation in industries such as aerospace, ship manufacturing, and energy and power. It aims to provide efficient and easy-to-use AI computing flow simulation software for industrial research engineers, university professors and students.

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindflow/docs/source_en/images/mindflow_archi_en.png" width="1200px" alt="" style="display: inline-block">

Code repository address: <https://gitee.com/mindspore/mindscience/tree/master/MindFlow>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation Deployment

   mindflow_install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Features

   features/solve_pinns_by_mindflow

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Physics-driven

   physics_driven/burgers1D
   physics_driven/darcy2D
   physics_driven/navier_stokes2D
   physics_driven/poisson_geometry
   physics_driven/taylor_green2D
   physics_driven/navier_stokes_inverse
   physics_driven/boltzmannD1V3
   physics_driven/kovasznay
   physics_driven/periodic_hill
   physics_driven/poisson_point_source

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Data-driven

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
   :caption: Data Mechanism Fusion

   data_mechanism_fusion/pde_net
   data_mechanism_fusion/percnn2d
   data_mechanism_fusion/percnn3d
   data_mechanism_fusion/phympgn

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: CFD-solver

   cfd_solver/lax_tube
   cfd_solver/sod_tube
   cfd_solver/couette
   cfd_solver/riemann2d
   cfd_solver/acoustic

.. toctree::
   :maxdepth: 1
   :caption: API Reference

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
