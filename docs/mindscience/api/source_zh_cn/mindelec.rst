MindElec
========

mindelec.architecture
---------------------

.. msplatformautosummary::
    :toctree: mindelec
    :nosignatures:
    :template: classtemplate.rst
 
	mindelec.architecture.FCSequential
	mindelec.architecture.InputScaleNet
	mindelec.architecture.LinearBlock
	mindelec.architecture.MTLWeightedLossCell
	mindelec.architecture.MultiScaleFCCell
	mindelec.architecture.ResBlock
	mindelec.architecture.get_activation

mindelec.common
---------------

.. msplatformautosummary::
    :toctree: mindelec
    :nosignatures:
    :template: classtemplate.rst
 
	mindelec.common.L2
	mindelec.common.LearningRate
	mindelec.common.get_poly_lr

mindelec.data
-------------

.. msplatformautosummary::
    :toctree: mindelec
    :nosignatures:
    :template: classtemplate.rst
 
	mindelec.data.BBoxType	
	mindelec.data.BoundaryBC
	mindelec.data.BoundaryIC
	mindelec.data.Dataset
	mindelec.data.Equation
	mindelec.data.ExistedDataConfig
	mindelec.data.ExistedDataset
	mindelec.data.MaterialConfig
	mindelec.data.PointCloud
	mindelec.data.PointCloudSamplingConfig
	mindelec.data.SamplingMode
	mindelec.data.StdPhysicalQuantity

mindelec.geometry
-----------------

.. msplatformautosummary::
    :toctree: mindelec
    :nosignatures:
    :template: classtemplate.rst
 
 	mindelec.geometry.create_config_from_edict
	mindelec.geometry.CSGDifference
	mindelec.geometry.CSGIntersection
	mindelec.geometry.CSGUnion
	mindelec.geometry.CSGXOR
	mindelec.geometry.Cuboid
	mindelec.geometry.Disk
	mindelec.geometry.Geometry
	mindelec.geometry.GeometryWithTime
	mindelec.geometry.HyperCube
	mindelec.geometry.Interval
	mindelec.geometry.PartSamplingConfig
	mindelec.geometry.Rectangle
	mindelec.geometry.SamplingConfig
	mindelec.geometry.TimeDomain


mindelec.loss
-------------

.. msplatformautosummary::
    :toctree: mindelec
    :nosignatures:
    :template: classtemplate.rst
 
	mindelec.loss.Constraints
	mindelec.loss.NetWithEval
	mindelec.loss.NetWithLoss
	mindelec.loss.get_loss_metric

mindelec.operators
------------------

.. msplatformautosummary::
    :toctree: mindelec
    :nosignatures:
    :template: classtemplate.rst
 
	mindelec.operators.Grad
	mindelec.operators.SecondOrderGrad

mindelec.solver
---------------

.. msplatformautosummary::
    :toctree: mindelec
    :nosignatures:
    :template: classtemplate.rst

	mindelec.solver.EvalCallback
	mindelec.solver.LossAndTimeMonitor
	mindelec.solver.Problem
	mindelec.solver.Solver

mindelec.vision
----------------

.. msplatformautosummary::
    :toctree: mindelec
    :nosignatures:
    :template: classtemplate.rst
 
	mindelec.vision.MonitorEval
	mindelec.vision.MonitorTrain
	mindelec.vision.image_to_video
	mindelec.vision.plot_eh
	mindelec.vision.plot_s11
	mindelec.vision.print_graph_1d
	mindelec.vision.print_graph_2d
	mindelec.vision.vtk_structure
