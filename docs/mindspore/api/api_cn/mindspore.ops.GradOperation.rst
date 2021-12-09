mindspore.ops.GradOperation
============================

.. py:class:: mindspore.ops.GradOperation(get_all=False, get_by_list=False, sens_param=False)

   一个高阶函数，用于为输入函数生成梯度函数。

   由 *GradOperation* 高阶函数生成的梯度函数可以通过构造参数自定义。

   构建一个以x和y为输入的函数 *net = Net()* ，并带有一个参数z，详见样例中的 *Net* 。

   生成一个梯度函数，该函数返回关于第一个输入的梯度（见样例中的 *GradNetWrtX* ）。

   1. 构建一个带有默认参数的 *GradOperation* 高阶函数： *grad_op = GradOperation()* 。

   2. 将 *net* 作为参数调用 *grad_op* ，得到梯度函数： *gradient_function = grad_op(net)* 。

   3. 用 *net* 的输入作为参数调用梯度函数，得到关于第一个输入的梯度：*grad_op(net)(x, y)* 。

   生成一个梯度函数，该函数返回关于所有输入的梯度（见样例中的 *GradNetWrtXY* ）。

   1. 构造一个带有 *get_all=True* 参数的 *GradOperation* 高阶函数，表示获得在样例中 *Net()* 中的x和y所有输入的梯度：*grad_op = GradOperation(get_all=True)* 。
   2. 将 *net* 作为参数调用 *grad_op* ，得到梯度函数： *gradient_function = grad_op(net)* 。
   3. 用 *net* 的输入作为参数调用梯度函数，得到所有输入的梯度：*gradient_function(x, y)* 。

   生成一个梯度函数，该函数返回关于给定参数的梯度（见样例中的 *GradNetWithWrtParams* ）。

   1. 构造一个带有 *get_by_list=True* 参数的GradOperation高阶函数： grad_op = GradOperation(get_by_list=True)。

   2. 当构建 *GradOperation* 高阶函数时，创建一个 *ParameterTuple* 和 *net* 作为参数输入， *ParameterTuple* 作为参数过滤器决定返回哪个梯度：*params = ParameterTuple(net.trainingable_params())* 。

   3. 将 *net* 和 *params* 作为参数输入 *grad_op* ，得到梯度函数： *gradient_function = grad_op(net, params)* 。

   4. 用 *net* 的输入作为参数调用梯度函数，得到关于给定参数的梯度： *gradient_function(x, y)* 。

   生成一个梯度函数，该函数以((dx, dy), (dz))的格式返回关于所有输入和给定参数的梯度（见样例中的 *GradNetWrtInputsAndParams* ）。

   1. 构建一个带有 *get_all=True* 和 *get_by_list=True* 参数的 *GradOperation* 高阶函数：*grad_op = GradOperation(get_all=True, get_by_list=True)* 。

   2. 当构建 *GradOperation* 高阶函数时，创建一个 *ParameterTuple* 和 *net* 作为参数输入：*params = ParameterTuple(net.trainingable_params())* 。

   3. 将 *net* 和 *params* 作为参数输入 *grad_op* ，得到梯度函数： *gradient_function = grad_op(net, params)* 。

   4. 用 *net* 的输入作为参数调用梯度函数，得到关于所有输入和给定参数的梯度：*gradient_function(x, y)* 。

   我们可以设置 *sens_param* 等于True来配置灵敏度（关于输出的梯度），向梯度函数传递一个额外的灵敏度输入值。这个输入值必须与 *net* 的输出具有相同的形状和类型（见样例中的 *GradNetWrtXYWithSensParam* ）。

   1. 构建一个带有 *get_all=True* 和 *sens_param=True* 参数的 *GradOperation* 高阶函数：*grad_op = GradOperation(get_all=True, sens_param=True)* 。

   2. 当 *sens_param=True* ，定义 *grad_wrt_output* （关于输出的梯度）：*grad_wrt_output = Tensor(np.ones([2, 2]).astype(np.float32))* 。

   3. 用 *net* 作为参数输入 *grad_op* ，得到梯度函数：*gradient_function = grad_op(net)* 。

   4. 用 *net* 的输入和 *sens_param* 作为参数调用梯度函数，得到关于所有输入的梯度：*gradient_function(x, y, grad_wrt_output)* 。

   **参数** ：

      - **get_all** (`bool`) –  如果等于True，获得所有输入的梯度。默认值：False。
      - **get_by_list** (`bool`) -  如果等于True，获得所有参数变量的梯度。如果 *get_all* 和 *get_by_list* 都等于False，则得到第一个输入的梯度。如果 *get_all* 和 *get_by_list* 都等于True，则同时得到关于输入和参数变量的梯度，输出形式为((关于输入的梯度)，(关于参数变量的梯度))。默认值：False。
      - **sens_param** (`bool`) -  是否在输入中配置灵敏度（关于输出的梯度）。如果sens_param等于False，自动添加一个 `ones_like(output)` 灵敏度。默认值：False。如果sensor_param等于True，灵敏度（关于输出的梯度），必须通过location参数或key-value pair参数来传递，如果是通过key-value pair参数传递value，那么key必须为sens。

   **返回** ：

      将一个函数作为参数，并返回梯度函数的高阶函数。

   **异常** ：

      **TypeError** - 如果 *get_all* ，*get_by_list* 或者 *sens_params* 不是布尔值。

   **支持平台** ：

      `Ascend` `GPU` `CPU`

   **样例** :

      .. code-block::

              >>> from mindspore import ParameterTuple
              >>> class Net(nn.Cell):
              >>>     def __init__(self):
              >>>         super(Net, self).__init__()
              >>>         self.matmul = P.MatMul()
              >>>         self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')
              >>>     def construct(self, x, y):
              >>>         x = x * self.z
              >>>         out = self.matmul(x, y)
              >>>         return out
              >>> 
              >>> class GradNetWrtX(nn.Cell):
              >>>     def __init__(self, net):
              >>>         super(GradNetWrtX, self).__init__()
              >>>         self.net = net
              >>>         self.grad_op = GradOperation()
              >>>     def construct(self, x, y):
              >>>         gradient_function = self.grad_op(self.net)
              >>>         return gradient_function(x, y)
              >>> 
              >>> x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
              >>> y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
              >>> output = GradNetWrtX(Net())(x, y)
              >>> print(output)
              [[1.4100001 1.5999999 6.6      ] 
              [1.4100001 1.5999999 6.6      ]]
              >>> 
              >>> class GradNetWrtXY(nn.Cell):
              >>>     def __init__(self, net):
              >>>         super(GradNetWrtXY, self).__init__()
              >>>         self.net = net
              >>>         self.grad_op = GradOperation(get_all=True)
              >>>     def construct(self, x, y):
              >>>         gradient_function = self.grad_op(self.net)
              >>>         return gradient_function(x, y)
              >>> 
              >>> x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
              >>> y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
              >>> output = GradNetWrtXY(Net())(x, y)
              >>> print(output)
              (Tensor(shape=[2, 3], dtype=Float32, value=
              [[ 4.50999975e+00,  2.70000005e+00,  3.60000014e+00],
              [ 4.50999975e+00,  2.70000005e+00,  3.60000014e+00]]), Tensor(shape=[3, 3], dtype=Float32, value= 
              [[ 2.59999990e+00,  2.59999990e+00,  2.59999990e+00], 
              [ 1.89999998e+00,  1.89999998e+00,  1.89999998e+00], 
              [ 1.30000007e+00,  1.30000007e+00,  1.30000007e+00]])) 
              >>> 
              >>> class GradNetWrtXYWithSensParam(nn.Cell):
              >>>     def __init__(self, net):
              >>>         super(GradNetWrtXYWithSensParam, self).__init__()
              >>>         self.net = net
              >>>         self.grad_op = GradOperation(get_all=True, sens_param=True)
              >>>         self.grad_wrt_output = Tensor([[0.1, 0.6, 0.2], [0.8, 1.3, 1.1]], dtype=mstype.float32)
              >>>     def construct(self, x, y):
              >>>         gradient_function = self.grad_op(self.net)
              >>>         return gradient_function(x, y, self.grad_wrt_output)
              >>> 
              >>> x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
              >>> y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
              >>> output = GradNetWrtXYWithSensParam(Net())(x, y)
              >>> print(output)
              (Tensor(shape=[2, 3], dtype=Float32, value=
              [[ 2.21099997e+00,  5.09999990e-01,  1.49000001e+00], 
              [ 5.58800030e+00,  2.68000007e+00,  4.07000017e+00]]), Tensor(shape=[3, 3], dtype=Float32, value= 
              [[ 1.51999998e+00,  2.81999993e+00,  2.14000010e+00], 
              [ 1.09999990e+00,  2.04999995e+00,  1.54999995e+00], 
              [ 9.00000036e-01,  1.54999995e+00,  1.25000000e+00]])) 
              >>> 
              >>> class GradNetWithWrtParams(nn.Cell):
              >>>     def __init__(self, net):
              >>>         super(GradNetWithWrtParams, self).__init__()
              >>>         self.net = net
              >>>         self.params = ParameterTuple(net.trainable_params())
              >>>         self.grad_op = GradOperation(get_by_list=True)
              >>>     def construct(self, x, y):
              >>>         gradient_function = self.grad_op(self.net, self.params)
              >>>         return gradient_function(x, y)
              >>> 
              >>> x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
              >>> y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)
              >>> output = GradNetWithWrtParams(Net())(x, y)
              >>> print(output)
              (Tensor(shape=[1], dtype=Float32, value= [ 2.15359993e+01]),)
              >>> 
              >>> class GradNetWrtInputsAndParams(nn.Cell):
              >>>     def __init__(self, net):
              >>>         super(GradNetWrtInputsAndParams, self).__init__()
              >>>         self.net = net
              >>>         self.params = ParameterTuple(net.trainable_params())
              >>>         self.grad_op = GradOperation(get_all=True, get_by_list=True)
              >>>     def construct(self, x, y):
              >>>         gradient_function = self.grad_op(self.net, self.params)
              >>>         return gradient_function(x, y)
              >>> 
              >>> x = Tensor([[0.1, 0.6, 1.2], [0.5, 1.3, 0.1]], dtype=mstype.float32)
              >>> y = Tensor([[0.12, 2.3, 1.1], [1.3, 0.2, 2.4], [0.1, 2.2, 0.3]], dtype=mstype.float32)
              >>> output = GradNetWrtInputsAndParams(Net())(x, y)
              >>> print(output)
              ((Tensor(shape=[2, 3], dtype=Float32, value=
              [[ 3.51999998e+00,  3.90000010e+00,  2.59999990e+00],
              [ 3.51999998e+00,  3.90000010e+00,  2.59999990e+00]]), Tensor(shape=[3, 3], dtype=Float32, value=
              [[ 6.00000024e-01,  6.00000024e-01,  6.00000024e-01],
              [ 1.89999998e+00,  1.89999998e+00,  1.89999998e+00],
              [ 1.30000007e+00,  1.30000007e+00,  1.30000007e+00]])), (Tensor(shape=[1], dtype=Float32, value=
              [ 1.29020004e+01]),))
              
              
              
              
              
              
              
              