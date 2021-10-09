mindspore.nn.Cell
==================

.. py:class:: mindspore.nn.Cell(auto_prefix=True, flags=None)

   所有神经网络的基类。
   
   一个 `Cell` 可以是一个单一的神经网络单元，如conv2d， relu， batch_norm等，也可以是组成网络的 `Cell` 的结合体。

   .. note:: 一般情况下，自动微分 (AutoDiff) 算法会自动生成梯度函数的实现，但是如果实现了自定义的反向传播函数 (bprop method)，梯度函数将会被Cell中定义的反向传播函数代替。反向传播函数将会接收一个包含关于输出的损失梯度的张量 `dout` 和一个包含前向传播结果的张量 `out` 。反向传播函数需要计算关于输入的损失梯度，关于参数变量的损失函数目前还不支持。反向传播函数必须包含self参数。

   **参数** ：

      - **auto_prefix** (`Cell`) – 递归生成命名空间。默认值：True。
      - **flags** (`dict`) - 网络配置信息，目前用于网络和数据集的绑定。用户还可以通过该参数自定义网络属性。默认值：None。

   **支持平台**：

   ``Ascend`` ``GPU`` ``CPU``

   **样例** :

      .. code-block::

            >>> import mindspore.nn as nn
            >>> import mindspore.ops as ops
            >>> class MyCell(nn.Cell):
            ...    def __init__(self):
            ...        super(MyCell, self).__init__()
            ...        self.relu = ops.ReLU()
            ... 
            ...    def construct(self, x):
            ...        return self.relu(x)

   .. py:property:: bprop_debug

      获取单元格自定义反向传播调试是否已启用。
   
   .. py:method:: cast_param(param)

      在pynative模式下，根据自动混合精度的级别转换权重类型。

      该接口目前在自动混合精度的情况下使用，通常不需要显式使用。

      **参数**：

         **param** (`Parameter`) – Parameter类型，并且需要转换类型的权重将会被强制转换。

      **返回**：

         Parameter类型，是被自动转换类型后的输入参数。

   .. py:method:: cells()

      返回一个在网络上的迭代器。

      **返回**：

         Iteration类型，在输入网络上的所有的子组件。
   
   .. py:method:: cells_and_names(cells=None, name_prefix="")

      返回一个该网络中所有组件的迭代器，包括组件名称及其本身。

      **参数**：

         - **cell** (`str`) – 需要进行迭代的网络。默认值：None。
         - **name_prefix** (`str`) – 命名空间。默认值：‘’。

      **返回**：

         Iteration类型，在输入网络中所有的组件和相对应的名称。

      **样例** :

            .. code-block::

                  >>> n = Net()
                  >>> names = []
                  >>> for m in n.cells_and_names():
                  ...    if m[0]:
                  ...       names.append(m[0])
      
   .. py:method:: compile(*inputs)

      编译网络。

      **参数**：

         **inputs** (`tuple`) – Cell对象的输入。

   .. py:method:: compile_and_run(*inputs)

      编译并运行网络。

      **参数**：

         **inputs** (`tuple`) – Cell对象的输入。

      **返回**：

         Object类型，执行的结果。

   .. py:method:: construct(*inputs, **kwargs)

      定义要执行的计算逻辑。所有子类都必须重写此方法。

      **返回**：

         Tensor类型，返回计算结果。

   .. py:method:: exec_checkpoint_graph()

      执行保存检查点图的操作。

   .. py:method:: extend_repr()

      设置Cell的扩展表示形式。

      如果要输出个性化的扩展信息，请在您自己的网络中重新实现此方法。

   .. py:method:: generate_scope()

      为网络中的每个组件对象生成作用域。

   .. py:method:: get_func_graph_proto()

      返回图二进制原型。

   .. py:method:: get_parameters(expand=True)

      返回一个该网络中所有组件参数的迭代器。

      生成此网络的参数。如果 `expand` 为True，则生成该网络和所有组件的参数。

      **参数**：

         **expand** (`bool`) – 如果为true，则生成该网络和所有组件的参数。否则，只生成该网络的直接组件的参数。默认值：True。

      **返回**：

         Iteration类型，输入网络的所有参数。

      **样例** :

            .. code-block::

                  >>> n = Net()
                  >>> parameters = []
                  >>> for item in net.get_parameters():
                  ...    parameters.append(item)

   .. py:method:: get_scope()

      返回一个网络中一个组件对象的作用域。

      生成此网络的参数。如果 `expand` 为True，则生成该网络和所有组件的参数。

      **返回**：

         String类型，网络组件的作用域。

   .. py:method:: infer_param_pipeline_stage()

      推导Cell中属于当前stage的所有参数。

      .. note:: 

         - 如果一个参数不属于任何被设置为pipeline_stage的网络，那么这个参数应该使用add_pipeline_stage来添加它的pipeline_stage信息。
         - 如果参数P被 stageA 和 stageB 两个不同阶段的算子使用，那么参数P应该在使用infer_param_pipeline_stage之前使用P.add_pipeline_stage(stageA)和P.add_pipeline_stage(stageB)添加它的stage信息。

      **返回**：

         属于流水线并行当前stage的参数。
      
      **异常**：

         **RuntimeError** – 如果有不属于任何阶段的参数存在。

   .. py:method:: init_parameters_data(auto_parallel_mode=False)

      初始化网络中的所有参数并替换原始保存的参数。

      .. note:: 

         trainable_params()和其他类似的接口可能会在 `init_parameters_data` 之后返回不同的参数实例，请勿保存这些结果。

      **参数**：

         **auto_parallel_mode** (`bool`) – 是否在自动并行模式下运行。
      
      **返回**：

         Dict[Parameter, Parameter]类型，返回原始参数和被替换参数的字典。
         
   .. py:method:: insert_child_to_cell(child_name, child_cell)

      使用给定名称将一个子网络添加到当前网络。

      **参数**：

         - **child_name** (`str`) – 子网络名称。
         - **child_cell** (`Cell`) – 要插入的子网络。
      
      **异常**：

         - **KeyError** – 如果子网络的名称不正确或与其他子网络名称重复。
         - **TypeError** – 如果子网络Cell类型不正确。

   .. py:method:: insert_param_to_cell(param_name, param, check_name=True)

      向当前网络添加参数。

      将指定名称的参数插入网络。请参考 `mindspore.nn.Cell.__setattr__` 源代码中的用法。

      **参数**：

         - **param_name** (`str`) – 参数名称。
         - **param** (`Parameter`) – 要插入到单元格的参数。
         - **check_name** (`bool`) – 明确名称输入是否兼容。默认值：True。
      
      **异常**：

         - **KeyError** – 如果参数名称为空或包含点。
         - **TypeError** – 如果用户没有先调用init()。
         - **TypeError** – 如果参数的类型不是Parameter。

   .. py:method:: load_parameter_slice(params)

      根据并行策略获取tensor分片并替换原始参数。

      请参考 `mindspore.common._Executor.compile` 源代码中的用法。

      **参数**：

         **params** (`dict`) – 用于初始化数据图的参数字典。

   .. py:method:: name_cells()

      返回一个网络中所有子网络的迭代器。

      包括该网络名称和网络本身。

      **返回**：

         Dict[String, Cell]，网络中的所有子网络和相应的名称。

   .. py:property:: param_prefix

      参数前缀是当前网络的直接子参数的前缀。

   .. py:property:: parameter_layout_dict

      `parameter_layout_dict` 表示一个参数的张量布局，这种张量布局是由分片策略和分布式算子信息推断出来的。

   .. py:method:: parameters_and_names(name_prefix="", expand=True)

      返回一个网络参数上的迭代器。
      
      包括参数名称及其本身。

      **参数**：

         - **name_prefix** (`str`) – 命名空间。默认值：‘’。
         - **expand** (`bool`) – 如果为true，则生成该网络和所有子网络的参数。否则，只生成该网络的直接成员的参数。默认值：True。

      **返回**：

         Iteration类型，网络中的所有名称和相应参数。

      **样例** :

            .. code-block::

                  >>> n = Net()
                  >>> names = []
                  >>> for m in n.parameters_and_names():
                  ...    if m[0]:
                  ...       names.append(m[0])

   .. py:method:: parameters_dict(recurse=True)

      获取参数字典。
      
      获取此网络的参数字典。

      **参数**：

         **recurse** (`bool`) – 是否包含子网络参数。默认值：True。

      **返回**：

         OrderedDict类型，返回参数字典。

   .. py:method:: recompute(**kwargs)

      使网络重新计算。网络中的所有算子将被重新计算。网络中的所有算子将会被设置成重新计算的。如果一个算子的计算结果被输出到一些反向节点来进行梯度计算，且被设置成重新计算的，那么我们会在反向传播中重新计算它，而不去存储在前向传播中的中间激活层的计算结果。

      .. note:: 

         - 如果计算涉及到诸如随机化或全局变量之类的操作，那么目前还不能保证等价。
         - 如果该网络中算子的重新计算api也被调用，则该算子的重新计算模式受算子的重新计算api的约束。

      **参数**：

         - **mode** (`bool`) – 表示是否重新计算该网络。默认值：True。
         - **output_recompute** (`bool`) – 表示当mode为true时是否重新计算此网络的输出。当mode为false时，这个参数无效。默认值：False。
         - **mp_comm_recompute** (`bool`) – 表示网络内的模型并行通信算子是否以自动并行或半自动并行方式重新计算。默认值：True。

   .. py:method:: register_backward_hook(fn)

      设置网络反向hook函数。此函数仅在Pynative Mode下支持。

      .. note:: fn必须有如下代码定义。 `cell_name` 是已注册网络的名称。 `grad_input` 是传递给网络的梯度。 `grad_output` 是计算或者传递给下一个网络或者算子的梯度，这个梯度可以被修改或者返回。hook_fn(cell_name, grad_input, grad_output) -> Tensor or None.

      **参数**：

         **fn** (`function`) – 以梯度作为输入的hook函数。

   .. py:method:: remove_redundant_parameters()

      删除冗余参数。
      
      这个接口通常不需要显式调用。

   .. py:method:: set_acc(acc_type)

      为了提高网络性能，可以配置网络自动启用来加速算法库中的算法。

      如果 `acc_type` 不在算法库内，请通过算法库查看算法库中的算法。

      .. note:: 有些加速算法可能会影响网络的准确性，请慎重选择。

      **参数**：

         **acc_type** (`str`) – 加速算法。

      **返回**：

         Cell类型，网络本身。
      
      **异常**：

         **ValueError** – 如果 `acc_type` 不在算法库内。

   .. py:method:: set_auto_parallel()

      将网络设置为自动并行模式。

      .. note:: 如果一个网络需要使用自动并行或半自动并行模式来进行训练、评估或预测，则该接口需要由网络调用。

   .. py:method:: set_broadcast_flag(mode=True)

      将网络设置为data_parallel模式。
      
      可以使用给定的名称作为属性访问网络。

      **参数**：

         **mode** (`bool`) – 表示模型是否为data_parallel模式。默认值：True。

   .. py:method:: set_comm_fusion(fusion_type, recurse=True)

      为网络中的所有参数设置 `comm_fusion` 。请参考 `mindspore.common.parameter.comm_fusion` 的描述。

      如果 `acc_type` 不在算法库内，请通过算法库查看算法库中的算法。

      .. note:: 当调用multiply函数时，属性值将被覆盖。

      **参数**：

         - **fusion_type** (`int`) – `comm_fusion` 的值。
         - **recurse** (`bool`) – 是否设置子网络的可训练参数。默认值：True。

   .. py:method:: set_grad(requires_grad=True)

      设置网络标志为梯度。在pynative模式下，该参数指定网络是否需要梯度。如果为True，则在执行正向网络时，将生成需要计算梯度的反向网络。
      
      **参数**：

         **requires_grad** (`bool`) – 指定网络是否需要梯度，如果为True，网络将以pynative模式构建反向网络。默认值：True。

      **返回**：

         Cell类型，网络本身。

   .. py:method:: set_parallel_input_with_inputs(*inputs)

      通过并行策略切片输入张量，并将切片后的输入张量设置为 `_parallel_input_run` 。
      
      **参数**：

         **inputs** (`tuple`) – 构造方法的输入。

   .. py:method:: set_param_fl(push_to_server=False, pull_from_server=False, requires_aggr=True)

      设置参数与服务器交互的方式。
      
      **参数**：

         - **push_to_server** (`bool`) – 是否将参数推送到服务器。默认值：False。
         - **pull_from_server** (`bool`) – 是否从服务器提取参数。默认值：False。
         - **push_to_server** (`bool`) – 是否在服务器中聚合参数。默认值：True。

   .. py:method:: set_param_ps(recurse=True, init_in_server=False)

      设置可训练参数是否由参数服务器更新，以及是否在服务器上初始化可训练参数。

      .. note:: 只在运行的任务处于参数服务器模式时有效。

      **参数**：

         - **recurse** (`bool`) – 是否设置子网络的可训练参数。默认值：True。
         - **init_in_server** (`bool`) – 是否在服务器上初始化由参数服务器更新的可训练参数。默认值：False。

   .. py:method:: set_train(mode=True)

      将网络设置为训练模式。

      网络本身和所有子网络将被设置为训练模式。对于训练和预测具有不同结构的层(如 `BatchNorm` )，将通过这个属性区分分支。如果设置为True，则执行训练分支，否则执行另一个分支。
      
      **参数**：

         **mode** (`bool`) – 指定模型是否为训练模型。默认值：True。

      **返回**：

         Cell类型，网络本身。

   .. py:method:: to_float(dst_type)

      在网络和子网络的所有输入上添加强制转换，以使用特定的浮点类型运行。

      如果 `dst_type` 是 `mindspore.dtype.float16` ，网络的所有输入(包括作为常量的input， Parameter， Tensor)都会被强制转换为float16。请参考 `mindspore.train.amp.build_train_network` 的源代码中的用法。

      .. note:: 多个调用将覆盖。

      **参数**：

         **dst_type** (`mindspore.dtype`) – 强制转换网络以 `dst_type` 类型运行。 `dst_type` 可以是 `mindspore.dtype.float16` 或者  `mindspore.dtype.float32` 。

      **返回**：

         Cell类型，网络本身。
      
      **异常**：

         **ValueError** – 如果 `dst_type` 不是float32，也不是float16。

   .. py:method:: trainable_params(recurse=True)

      返回所有可训练参数。

      返回一个所有可训练参数的列表。

      **参数**：

         **recurse** (`bool`) – 是否包含子网络的可训练参数。默认值：True。

      **返回**：

         List类型，可训练参数列表。

   .. py:method:: untrainable_params(recurse=True)

      返回所有不可训练参数。

      返回一个所有不可训练参数的列表。

      **参数**：

         **recurse** (`bool`) – 是否包含子网络的不可训练参数。默认值：True。

      **返回**：

         List类型，不可训练参数列表。

   .. py:method:: update_cell_prefix()

      更新所有子网络的 `self.param_prefix` 。

      在被调用后，它可以通过 `_param_prefix` 获取网络的所有子网络的名称前缀。

   .. py:method:: update_cell_type(cell_type)

      当遇到量化感知训练网络时，更新当前网络类型。。

      在被调用后，它可以将单元格类型设置为 `cell_type` 。

   .. py:method:: update_parameters_name(prefix="", recurse=True)

      用给定的前缀字符串更新参数的名称。

      将给定的前缀添加到参数名称中。

      **参数**：

         - **prefix** (`str`) – 前缀字符串。默认值：''。
         - **recurse** (`bool`) – 是否包含子网络的参数。默认值：True。
     