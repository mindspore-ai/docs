Common Network Components
==========================

Overview
---------

MindSpore encapsulates some common network components for network training, inference, gradient calculation, and data processing.

These network components can be directly used by users and are also used in more advanced encapsulation APIs such as `model.train` and `model.eval`.

The following describes three network components, `GradOperation`, `WithLossCell`, and `TrainOneStepCell`, in terms of functions, usage, and internal use.

.. toctree::
   :maxdepth: 1

   gradoperation
   WithLossCell
   trainonestepcell
