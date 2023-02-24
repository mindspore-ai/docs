MindFlow Introduction
=====================

Flow simulation aims to solve the fluid governing equation under a given boundary condition by numerical methods, so as to realize the flow analysis, prediction and control. It is widely used in engineering design in aerospace, ship manufacturing, energy and power industries. The numerical methods of traditional flow simulation, such as finite volume method and finite difference method, are mainly implemented by commercial software, requiring physical modeling, mesh generation, numerical dispersion, iterative solution and other steps. The simulation process is complex and the calculation cycle is long. AI has powerful learning fitting and natural parallel inference capabilities, which can improve the efficiency of the flow simulation.

MindSpore Flow is a flow simulation suite developed based on MindSpore. It supports AI flow simulation in industries such as aerospace, ship manufacturing, and energy and power. It aims to provide efficient and easy-to-use AI computing flow simulation software for industrial research engineers, university professors and students.

.. raw:: html

   <img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/docs/mindflow/docs/source_en/images/mindflow_archi_en.png" width="600px" alt="" >


MindSpore AI Fluid Simulation Suite

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Installation Deployment

   mindflow_install

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Physics-driven

   physics_driven/burgers1D
   physics_driven/navier_stokes2D

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Data-driven

   data_driven/dfyf
   data_driven/FNO1D
   data_driven/FNO2D

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: CFD-solver

   cfd/lax_tube
   cfd/sod_tube
   cfd/couette
   cfd/riemann2d

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Data Mechanism Fusion

   data_mechanism_fusion/pde_net

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Features

   features/solve_pinns_by_mindflow

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   mindflow.cell
   mindflow.cfd
   mindflow.common
   mindflow.data
   mindflow.geometry
   mindflow.loss
   mindflow.operators
   mindflow.pde
   mindflow.solver

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Reference Document

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
