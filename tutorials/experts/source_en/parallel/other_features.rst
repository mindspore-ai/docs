Other Features
==============

.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0.0-alpha/resource/_static/logo_source_en.png
    :target: https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/tutorials/experts/source_en/parallel/other_features.rst

.. toctree::
  :maxdepth: 1
  :hidden:

  sharding_propagation
  parameter_server_training
  comm_fusion
  dataset_slice
  pynative_shard_function_parallel
  ms_operator

`Sharding Propagation <https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/sharding_propagation.html>`__
--------------------------------------------------------------------------------------------------------------------------

In operator-level parallelism, the user is required to configure a
slicing strategy for each operator in the forward network (if not
configured, the data-parallel policy is used by default). The slicing
strategy propagation feature can configure only a few operators to
automatically generate a feasible sharding strategy for operators
without a sharding strategy, and achieve the effect of minimizing
communication overhead.

`Parameter Server Training <https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/parameter_server_training.html>`__
-------------------------------------------------------------------------------------------------------------------------------------

Parameter Server is a widely used architecture in distributed training,
which has better flexibility, scalability, and node disaster tolerance
than the AllReduce training method of data parallel synchronization. The
parameter server supports both synchronous SGD (Stochastic Gradient
Descent) and asynchronous SGD training algorithms. In terms of
scalability, the calculation of the model and the update of the model
are deployed in the worker and server processes respectively, so that
the resources of the worker and server can be scaled horizontally
independently (adding or removing the worker and server resources). In
addition, in the environment of large-scale data centers, computing
equipment, networks and storage often have various failures that lead to
some node abnormalities, and under the architecture of parameter
servers, such failures can be easily handled without affecting the tasks
in training.

`Communication Operator Fusion <https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/comm_fusion.html>`__
--------------------------------------------------------------------------------------------------------------------------

In the distributed training scenario, cross-device or even cross-node
data transmission is a bottleneck that restricts scalability and
computing power utilization. Communication operator fusion is an
important method to improve the utilization of network resources and
accelerate the efficiency of data transmission, which packages the
communication operators of the same source node and the destination node
and executes them at the same time to avoid the additional overhead
caused by multiple single operator execution.

`Dataset Slicing <https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/dataset_slice.html>`__
--------------------------------------------------------------------------------------------------------------

When doing distributed training, you need to import the training dataset
to each device. There are two common ways to import: 1) Import in
parallel with the data, that is, the data is split into match
dimensions, and each device is imported as part; 2) Import full amount
of data per device. In addition, when some dimensions of the data are
particularly large (such as the H/W dimension of the remote sensing
picture may be particularly large), even if the sample size is small,
the picture needs to be split, that is, the data is split in the H/W
dimension, and each device reads a part of the picture. This special
performance supports splitting datasets into specific dimensions to meet
training requirements in the field of large-format image processing.

`Functional Operator Splitting <https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/pynative_shard_function_parallel.html>`__
------------------------------------------------------------------------------------------------------------------------------------------------

In dynamic graph mode, you specify that a part of the network structure
executes in graph mode and performs various parallel operations.

`Performing Distributed Training on K8S Clusters <https://www.mindspore.cn/tutorials/experts/en/r2.0.0-alpha/parallel/ms_operator.html>`__
--------------------------------------------------------------------------------------------------------------------------------------------

MindSpore Operator is a plugin that follows Kubernetes’ Operator pattern
(based on the CRD-Custom Resource Definition feature) and implements
distributed training on Kubernetes. MindSpore Operator defines
Scheduler, PS, worker three roles in CRD, and users can easily use
MindSpore on K8S for distributed training through simple YAML file
configuration. The code repository of mindSpore Operator is described
in: `ms-operator <https://gitee.com/mindspore/ms-operator/>`__.

Description of the Interface Related to the Feature
---------------------------------------------------

+-------------+----------------------+-------------------+-------------------+
| Feature     | Feature interface    | Description       | Function          |
|             |                      |                   |                   |
| category    |                      |                   |                   |
|             |                      |                   |                   |
+=============+======================+===================+===================+
| operat\     | shard(in_strategy\   | Set the sharding  | Reduce the memory |
| or          | =None,\              | strategy of the   | capacity of a     |
| parall\     | out_strategy=None)\  | input and output  | single device by  |
| el          | In Primitive class   | tensors of the    | slicing the       |
|             |                      | operator (where   | tensor involved   |
|             |                      | the sharding      | in each operator  |
|             |                      | strategy of the   | in the network    |
|             |                      | output tensor     | model to complete |
|             |                      | only supports     | the large model   |
|             |                      | some operators,   | training/inferenc |
|             |                      | such as Gauther   | e.                |
|             |                      | and MatMul.)      | Or use cluster    |
|             |                      |                   | resources to      |
|             |                      |                   | perform           |
|             |                      |                   | distributed       |
|             |                      |                   | computing to      |
|             |                      |                   | reduce the        |
|             |                      |                   | overall execution |
|             |                      |                   | time.             |
+-------------+----------------------+-------------------+-------------------+
|             | add_prim_attr(nam\   | Gather            | In the            |
|             | e, value)\           | Operator:add_prim\| recommended       |
|             | In Primitive class   | _attr(“manual_spl\| field, there is a |
|             |                      | it”,              | scene where each  |
|             |                      | config):          | column of the     |
|             |                      | Configure a\      | dataset           |
|             |                      | non-uniform\      | corresponds to a  |
|             |                      | sharding strategy | subtable. In this |
|             |                      | for its first     | scenario, using   |
|             |                      | input, where      | this              |
|             |                      | config type is    | configuration can |
|             |                      | tuple, which      | reduce traffic    |
|             |                      | describes how the | and improve       |
|             |                      | first parameter,  | overall           |
|             |                      | dimension 0, is   | performance.      |
|             |                      | split. For        |                   |
|             |                      | example , ( 10 ,\ |                   |
|             |                      | 20 , 30 , 4 )     |                   |
|             |                      | means that the    |                   |
|             |                      | 0th dimension of  |                   |
|             |                      | the first input   |                   |
|             |                      | of the operator   |                   |
|             |                      | is tangent into 4 |                   |
|             |                      | parts , and the   |                   |
|             |                      | shape size of     |                   |
|             |                      | each part is 10 , |                   |
|             |                      | 20 , 30 , 4,      |                   |
|             |                      | respectively.     |                   |
+-------------+----------------------+-------------------+-------------------+
|             |                      | EmbeddingLookUp   | In the            |
|             |                      | Operator:add_prim\| recommended       |
|             |                      | _attr(“primi\     | field, there is a |
|             |                      | tive_target”,     | particularly      |
|             |                      | “CPU”): Configure | large scene of    |
|             |                      | it to execute on  | the Embedding     |
|             |                      | the CPU for       | Table, in order   |
|             |                      | heterogeneous     | to save device    |
|             |                      | scenarios.        | memory, you can   |
|             |                      |                   | use this          |
|             |                      |                   | configuration to  |
|             |                      |                   | put               |
|             |                      |                   | EmbeddingLookUp   |
|             |                      |                   | on the CPU to     |
|             |                      |                   | execute to        |
|             |                      |                   | complete the      |
|             |                      |                   | training of the   |
|             |                      |                   | recommended large |
|             |                      |                   | model.            |
+-------------+----------------------+-------------------+-------------------+
|             | set_auto_parallel\   | Indicate whether  | AllToAll          |
|             | _context(enable_a\   | the AllToAll      | communication can |
|             | lltoall=bool_valu\   | communication     | reduce the amount |
|             | e)                   | operator is       | of communication  |
|             |                      | allowed to be     | data and improve  |
|             |                      | generated when    | communication     |
|             |                      | communicating,    | efficiency, but   |
|             |                      | and its value is  | it requires       |
|             |                      | the bool type,    | environmental     |
|             |                      | which defaults to | support.          |
|             |                      | False.            |                   |
+-------------+----------------------+-------------------+-------------------+
| Pipeline    | set_auto_parallel\   | Set the number of | Specify the       |
|             | _context(pipeline\   | pipes in pipeline | number of stages, |
| parallel    | _stages=stage_num)   | parallelism, the  | limiting the      |
|             |                      | value of which is | communication     |
|             |                      | a positive        | domain of the     |
|             |                      | integer, and the  | collection        |
|             |                      | value range is    | communication to  |
|             |                      | [1, number of     | the stage, and    |
|             |                      | devices].         | the               |
|             |                      |                   | point-to-point    |
|             |                      |                   | communication     |
|             |                      |                   | between the       |
|             |                      |                   | stages.           |
+-------------+----------------------+-------------------+-------------------+
|             | pipeline_stage(value\| Set which stage   | Set which stage   |
|             | ) In Cell class      | the Cell executes | the Cell executes |
|             |                      | in.               | in.               |
+-------------+----------------------+-------------------+-------------------+
|             | PipelineCell(netw\   | Specify the       | Specify           |
|             | ork, micro_size)     | number of         | micro_size can    |
|             |                      | MicroSizes for    | reduce the idle   |
|             |                      | the training      | wait time between |
|             |                      | network, where    | stages and        |
|             |                      | the network is    | improve the       |
|             |                      | the network to be | overall           |
|             |                      | trained and the   | efficiency of     |
|             |                      | micro_size is a   | pipeline          |
|             |                      | positive integer. | parallel.         |
+-------------+----------------------+-------------------+-------------------+
| Optimizer   | set_auto_parallel\   | Indicate whether  | Optimizer         |
|             | _context(enable_p\   | optimizer         | parallel saves    |
| parallel    | arallel_optimizer\   | parallelism is    | static memory     |
|             | =bool_value)         | enabled. Its      | overhead, but     |
|             |                      | value is bool     | increases         |
|             |                      | type, and the     | communication     |
|             |                      | default is False. | overhead.         |
+-------------+----------------------+-------------------+-------------------+
|             | set_auto_parallel\   | This              | gradient_accumula\|
|             | _context(parallel\   | configuration     | tion_shard        |
|             | _optimizer_config\   | takes effect only | true saves a      |
|             | =config)             | after optimizer   | portion of the    |
|             |                      | parallel is       | parameter size of |
|             |                      | turned on. The    | static memory,    |
|             |                      | config is a dict  | but increases     |
|             |                      | and supports two  | communication     |
|             |                      | key values:       | overhead.         |
|             |                      | gradient_accumula\| Optimizer         |
|             |                      | tion_shard(bool): | sharding          |
|             |                      | If True, the      | thresholds allow  |
|             |                      | cumulative        | smaller shape     |
|             |                      | gradient variable | parameters to be  |
|             |                      | will be sharded   | not optimized for |
|             |                      | on the data       | splitting to save |
|             |                      | parallelism,      | communication     |
|             |                      | defaulting to     | resources.        |
|             |                      | False.parallel_op\|                   |
|             |                      | timizer_threshold\|                   |
|             |                      | (int):            |                   |
|             |                      | This value        |                   |
|             |                      | represents the    |                   |
|             |                      | optimizer         |                   |
|             |                      | sharding          |                   |
|             |                      | threshold in KB   |                   |
|             |                      | (default value is |                   |
|             |                      | 64KB). When the   |                   |
|             |                      | parameter size    |                   |
|             |                      | does not exceed\  |                   |
|             |                      | this value, it    |                   |
|             |                      | will not be       |                   |
|             |                      | split.            |                   |
+-------------+----------------------+-------------------+-------------------+
| Recompute   | recompute(mode=True)\| Used to specify   | After enabling    |
|             | In primitive class   | whether the       | operator          |
|             |                      | operator needs to | recalculation,    |
|             |                      | be recalculated,  | you can reduce    |
|             |                      | and its value is  | the peak of       |
|             |                      | bool type, which  | dynamic memory,   |
|             |                      | defaults to True  | but increase the  |
|             |                      | and means that    | overall           |
|             |                      | the operator      | computation       |
|             |                      | recalculation is  | amount.           |
|             |                      | enabled.          |                   |
+-------------+----------------------+-------------------+-------------------+
|             | recompute(\*\*kwargs\| When this         | Enable Cell       |
|             | ) In Cell class      | interface is      | recompute and     |
|             |                      | called, the       | configure whether |
|             |                      | operator in this  | the model         |
|             |                      | Cell is           | parallel          |
|             |                      | recalculated.The  | communication     |
|             |                      | input parameter   | operator and the  |
|             |                      | has two bool      | optimizer         |
|             |                      | class             | parallel          |
|             |                      | options:mp_comm_r\| communication     |
|             |                      | ecompute:         | operator are      |
|             |                      | Whether to enable | recomputed. When  |
|             |                      | model parallel    | the communication |
|             |                      | communication     | operator is       |
|             |                      | operator          | recomputed, it    |
|             |                      | recalculation,    | consumes          |
|             |                      | and the default   | communication     |
|             |                      | is                | resources but     |
|             |                      | True.parallel_opt\| reduces the peak  |
|             |                      | imizer_comm_recom\| of dynamic        |
|             |                      | pute:             | memory.           |
|             |                      | Whether to enable |                   |
|             |                      | optimizer         |                   |
|             |                      | parallel          |                   |
|             |                      | communication     |                   |
|             |                      | operator          |                   |
|             |                      | recompute, and    |                   |
|             |                      | the default is    |                   |
|             |                      | False.            |                   |
+-------------+----------------------+-------------------+-------------------+

