# CANN常见错误分析

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_zh_cn/debug/cann_error_cases.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>&nbsp;&nbsp;

本文主要介绍用户常见的CANN错误处理方法。在遇到CANN错误时，MindSpore的日志可能不足以分析相关错误，可以通过设置以下两个环境变量来打印CANN的日志以更好地分析错误：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1        # CANN日志级别，0为debug，1为info，2为warning，3为error
export ASCEND_SLOG_PRINT_TO_STDOUT=1    # 配置开启日志打屏
```

另外，如果需要知道图编译过程中的网络结构和算子信息，可以[查看中间文件](https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/mindir.html)，保存IR图等中间文件进行查看。

## AICORE算子编译问题

AICORE算子编译错误根据不同模块会以`E5`~`EB`开头，其中`E5`~`E8`错误码AICORE算子编译过程中一些校验报错，用户可以根据报错信息先尝试自行检查修正，而`E9`~`EB`错误码是TBE编译前后端报的错，一般来说报错就意味着算子规格TBE还不支持，可以从报错日志中获取具体信息。下面可以看一些具体的AICORE算子编译失败问题：

### E80000: StridedSliceGradD输入值非法

```c++
[WARNING] CORE(51545,ffff8ba74480,python):2019-07-25-19:17:35.411.770 [mindspore/core/ir/anf_extends.cc:66] fullname_with_scope] Input 0 of cnode is not a value node, its type is CNode.
[WARNING] DEVICE(51545,ffff8ba74480,python):2019-07-25-19:17:41.850.324 [mindspore/ccsrc/runtime/hardware/ascend/ascend_graph_optimization.cc:255] SelectKernel] There are 2 node/nodes used raise precision to selected the kernel!
[CRITICAL] KERNEL(51545,ffff8ba74480,python):2019-07-25-19:17:54.525.980 [mindspore/ccsrc/backend/kernel_compiler/tbe/tbe_kernel_compile.cc:494] QueryProcess] Single op compile failed, op: strided_slice_grad_d_1157509189447431479_0
except_msg: 2019-07-25 19:17:54.522852: Query except_msg:Traceback (most recent call last):
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/te_fusion/parallel_compilation.py", line 1453, in run
    op_name=self._op_name)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/te_fusion/fusion_manager.py", line 1283, in build_single_op
    compile_info = call_op()
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/te_fusion/fusion_manager.py", line 1271, in call_op
    opfunc(*inputs, *outputs, *new_attrs, **kwargs)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/tbe/common/utils/para_check.py", line 545, in _in_wrapper
    return func(*args, **kwargs)
  File "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/impl/strided_slice_grad_d.py", line 1101, in strided_slice_grad_d
    _check_shape_parameter(shape, shape_dy, begin_shape, end_shape, stride_shape)
  File "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe/impl/strided_slice_grad_d.py", line 996, in _check_shape_parameter
    "1", str(strides_i))
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/te/utils/error_manager/error_manager_vector.py", line 40, in raise_err_input_value_invalid
    return raise_err_input_value_invalid(op_name, param_name, excepted_value, real_value)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/tbe/common/utils/errormgr/error_manager_vector.py", line 41, in raise_err_input_value_invalid
    raise RuntimeError(args_dict, msg)
RuntimeError: ({'errCode': 'E80000', 'op_name': 'strided_slice_grad_d', 'param_name': 'strides[0]', 'excepted_value': '1', 'real_value': '2'}, 'In op[strided_slice_grad_d], the parameter[strides[0]] should be , but actually is [2].')

The function call stack:
In file /home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/ops/_grad/grad_array_ops.py(700)/        dx = input_grad(dout, x_shape, begin, end, strides)/
Corresponding forward node candidate:
- In file /home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/ops/composite/multitype_ops/_compile_utils.py(306)/        return P.StridedSlice(begin_mask, end_mask, 0, 0, 0)(data, begin_strides, end_strides, step_strides)/
  In file /home/jenkins/models/official/cv/lenet/src/lenet.py(61)/        y = x[0::2] #切分操作，x.shape=(32,10) y.shape=(32),导致降维/
  In file /home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/amp.py(126)/            out = self._backbone(data)/
  In file /home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/wrap/loss_scale.py(332)/        loss = self.network(*inputs)/
  In file /home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/dataset_helper.py(95)/        return self.network(*outputs)/
```

从上述错误码的报错信息中可以看到，StridedSliceGradD算子的参数strides中的第一个数据预期应该是1，但是实际得到的是2，故而报错。用户可以根据IR图来确认出错的算子信息，并修正strides参数。

### E80012: ReduceSum算子输入维度过高

```c++
RuntimeError: ({'errCode': 'E80012', 'op_name': 'reduce_sum_d', 'param_name': 'x', 'min_value': 0, 'max_value': 8, 'real_value': 10}, 'In op, the num of dimensions of input/output[x] should be in the range of [0, 8], but actually is [10].')
```

从上述错误码的报错信息中可以看到，ReduceSum算子的输入和输出最高只支持8维的数据，但是实际遇到了10维的数据，故而报错。用户可以通过修改网络脚本来避免对ReduceSum输入10维的数据，从而规避这个错误。

### E80029: Assign算子shape不一致问题

```c++
RuntimeError: ({'errCode': 'E80029', 'op_name': 'assign', 'param_name1': 'ref', 'param_name2': 'value', 'error_detail': 'Shape of ref and value should be same'}, 'In op[assign], the shape of inputs[ref][value] are invalid, [Shape of ref and value should be same].')
```

Assign算子的逻辑是使用第二个输入（即value）对第一个输入的parameter（即ref）进行赋值操作，从上述错误码的报错信息中可以看到，这两个输入预期shape应该一致，但是实际并不一致，故而报错。用户可以根据IR图来确认出错的算子，并修正Assign算子的输入。

### EB0000: Transpose规格不支持

```c++
[ERROR] KERNEL(1062,fffe557fa160,python):2021-10-11-22:37:53.881.210 [mindspore/ccsrc/backend/kernel_compiler/tbe/tbe_kernel_parallel_build.cc:99] TbeOpParallelBuild] task compile Failed, task id:3, cause:TBEException:ERROR:

Traceback (most recent call last):
  File "/usr/local/python3.7/lib/python3.7/site-packages/mindspore/_extends/parallel_compile/tbe_compiler/compiler.py", line 155, in build_op
    optune_opt_list=op_tune_list)
  File "/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/te_fusion/fusion_manager.py", line 1258, in build_single_op
    compile_info = call_op()
  File "/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/te_fusion/fusion_manager.py", line 1246, in call_op
    opfunc(*inputs, *outputs, *attrs, **kwargs)
  File "/usr/local/Ascend/nnae/latest/fwkacllib/python/site-packages/tbe/common/utils/para_check.py", line 539, in _in_wrapper
    return func(*args, **kwargs)
  File "/usr/local/Ascend/nnae/latest/opp/op_impl/built-in/ai_core/tbe/impl/transpose_d.py", line 24644, in transpose_d
    tvm.build(sch, tensor_list, "cce", name=kernel_name)
  File "/usr/local/Ascend/nnae/5.0.T301/fwkacllib/python/site-packages/te/tvm/build_module.py", line 799, in build
    build_cce(inputs, args, target, target_host, name, rules, binds, evaluates)
  File "/usr/local/Ascend/nnae/5.0.T301/fwkacllib/python/site-packages/te/tvm/cce_build_module.py", line 855, in build_cce
    cce_lower(inputs, args, name, binds=binds, evaluates=evaluates, rule=rules)
  File "/usr/local/Ascend/nnae/5.0.T301/fwkacllib/python/site-packages/te/tvm/cce_build_module.py", line 50, in wrapper
    r = fn(*args, **kw)
  File "/usr/local/Ascend/nnae/5.0.T301/fwkacllib/python/site-packages/te/tvm/cce_build_module.py", line 404, in cce_lower
    arg_list, cfg, evaluates, inputs_num, rule)
  File "/usr/local/Ascend/nnae/5.0.T301/fwkacllib/python/site-packages/te/tvm/cce_build_module.py", line 146, in cce_static_lower
    stmt = ir_pass.StorageRewriteCCE(stmt)
  File "/usr/local/Ascend/nnae/5.0.T301/fwkacllib/python/site-packages/te/tvm/_ffi/_ctypes/function.py", line 209, in __call__
    raise get_last_ffi_error()
tvm._ffi.base.TVMError: Traceback (most recent call last):
  [bt] (5) /usr/local/Ascend/nnae/5.0.T301/fwkacllib/lib64/libtvm.so(TVMFuncCall+0x70) [0xffffa0ccd5a8]
  [bt] (4) /usr/local/Ascend/nnae/5.0.T301/fwkacllib/lib64/libtvm.so(std::_Function_handler<void (tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*), void tvm::runtime::TypedPackedFunc<tvm::Stmt (tvm::Stmt)>::AssignTypedLambda<tvm::Stmt (*)(tvm::Stmt)>(tvm::Stmt (*)(tvm::Stmt))::{lambda(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)#1}>::_M_invoke(std::_Any_data const&, tvm::runtime::TVMArgs&&, tvm::runtime::TVMRetValue*&&)+0x5c) [0xffff9ffb69fc]
  [bt] (3) /usr/local/Ascend/nnae/5.0.T301/fwkacllib/lib64/libtvm.so(tvm::ir::StorageRewriteCCE(tvm::Stmt)+0x590) [0xffffa09753d0]
  [bt] (2) /usr/local/Ascend/nnae/5.0.T301/fwkacllib/lib64/libtvm.so(+0x1d40658) [0xffffa0974658]
  [bt] (1) /usr/local/Ascend/nnae/5.0.T301/fwkacllib/lib64/libtvm.so(+0x1d61844) [0xffffa0995844]
  [bt] (0) /usr/local/Ascend/nnae/5.0.T301/fwkacllib/lib64/libtvm.so(tvm_dmlc::LogMessageFatal::~LogMessageFatal()+0x4c) [0xffff9ff6602c]
  File "storage_rewrite_cce.cc", line 840

TVMError: [EB0000] Check failed: need_nbits <= info->max_num_bits (4194304 vs. 2097152) : data_ub exceed bound of memory: local.UB ! Current IR Stmt:copy_ubuf_to_gm(tvm_access_ptr(0h, res, ((i0*263168) + (blockIdx.x*1024)), 1024, 2, 1, 2048, 1, 2048, 2048), tvm_access_ptr(0h, data_ub, (i0*1024), 1024, 1, 1, 2048, 1, 2048, 2048), 0, 1, 64, 0, 0)

input_args: {"SocInfo": {"autoTilingMode": "NO_TUNE", "coreNum": "", "coreType": "", "l1Fusion": "false", "l2Fusion": "false", "l2Mode": "2", "op_debug_level": "", "op_impl_mode": "", "op_impl_mode_list": [], "socVersion": "Ascend910A"}, "impl_path": "", "op_info": {"Type": "Transpose", "attr_desc": [[0, 2, 1, 3]], "attrs": [{"name": "perm", "valid": true, "value": [0, 2, 1, 3]}], "full_name": "Default/encode_image-EncodeImage/visual-ViT/body-Transformer/layers-SequentialCell/0-SequentialCell/0-ResidualCell/cell-SequentialCell/1-AttentionWithMask/Transpose-op662", "gen_model": "single", "graph_id": 0, "inputs": [[{"addr_type": 0, "dtype": "float16", "format": "NCHW", "name": "x_0", "ori_format": "NCHW", "ori_shape": [256, 16, 257, 64], "param_type": "required", "range": [[256, 256], [16, 16], [257, 257], [64, 64]], "shape": [256, 16, 257, 64], "valid": true}]], "is_dynamic_shape": false, "kernel_name": "Transpose_4779120815397556904_6", "module_name": "impl.transpose_d", "name": "transpose_d", "op_tune_list": "ALL", "op_tune_switch": "on", "outputs": [[{"addr_type": 0, "dtype": "float16", "format": "NCHW", "name": "y", "ori_format": "NCHW", "ori_shape": [256, 257, 16, 64], "param_type": "required", "range": [[256, 256], [257, 257], [16, 16], [64, 64]], "shape": [256, 257, 16, 64], "valid": true}]], "pass_list": "ALL", "py_module_path": "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe", "rl_tune_list": "ALL", "rl_tune_switch": "on", "socVersion": "Ascend910A"}, "platform": "TBE", "reset_op_info": [{"type": "clear_vector", "bin_path": "./kernel_meta/vector_random_buff.o", "kernel_name": "vector_random_buff"}, {"type": "clear_cube", "bin_path": "./kernel_meta/cube_random_buff.o", "kernel_name": "cube_random_buff"}]} trace:
In file /cache/user-job-dir/mindspore_clip_20210923/msvision/backbone/clip.py(380)/        out = self.transpose(out, (0, 2, 1, 3))/
In file /usr/local/python3.7/lib/python3.7/site-packages/mindspore/nn/layer/container.py(236)/        for cell in self.cell_list:/
In file /cache/user-job-dir/mindspore_clip_20210923/msvision/backbone/vit.py(105)/        return self.cell(x, **kwargs) + x/
In file /usr/local/python3.7/lib/python3.7/site-packages/mindspore/nn/layer/container.py(236)/        for cell in self.cell_list:/
In file /usr/local/python3.7/lib/python3.7/site-packages/mindspore/nn/layer/container.py(236)/        for cell in self.cell_list:/
In file /cache/user-job-dir/mindspore_clip_20210923/msvision/backbone/vit.py(427)/        return self.layers(x)/
In file /cache/user-job-dir/mindspore_clip_20210923/msvision/backbone/vit.py(232)/        x = self.body(x)/
In file /cache/user-job-dir/mindspore_clip_20210923/msvision/backbone/clip.py(562)/        return self.visual(image)/
In file /cache/user-job-dir/mindspore_clip_20210923/msvision/backbone/clip.py(635)/        image_features = self.encode_image(image)/
```

从上述错误码的报错信息中可以看到，这个transpose算子的规格不支持，其做内存拷贝指令时数据超过了限制。针对此类问题，用户可以到[MindSpore社区](https://gitee.com/mindspore)提交issue获取帮助。

## AICORE算子执行问题

### EZ9999: AICORE算子执行失败

一般AICORE算子执行失败会报`EZ9999`错误，同时MindSpore侧会有`Call rt api rtStreamSynchronize failed`的报错日志，根据错误码日志可能明确执行失败的算子，如下述报错场景是Add算子执行失败：

```c++
[EXCEPTION] GE(118114,ffff4effd1e0,python):2021-09-11-16:48:31.511.063 [mindspore/ccsrc/runtime/device/ascend/ge_runtime/runtime_model.cc:233] Run] Call rt api rtStreamSynchronize failed, ret: 507011
[WARNING] DEVICE(118114,ffff4effd1e0,python):2021-09-11-16:48:31.511.239 [mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc:662] GetDumpPath] MS_OM_PATH is null, so dump to process local path, as ./rank_id/node_dump/...
[ERROR] DEVICE(118114,ffff4effd1e0,python):2021-09-11-16:48:31.511.290 [mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc:679] DumpTaskExceptionInfo] Task fail infos task_id: 13, stream_id: 14, tid: 118198, device_id: 0, retcode: 507011 ( model execute failed)
[ERROR] DEVICE(118114,ffff4effd1e0,python):2021-09-11-16:48:31.511.460 [mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc:688] DumpTaskExceptionInfo] Dump node (Default/Add-op8) task error input/output data to: ./rank_0/node_dump trace:
In file /home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/ops/composite/multitype_ops/add_impl.py(212)/    return F.add(x, y)/
In file try.py(19)/        z = x + z /
[EXCEPTION] SESSION(118114,ffff4effd1e0,python):2021-09-11-16:48:31.524.043 [mindspore/ccsrc/backend/session/ascend_session.cc:1456] Execute] run task error!
[ERROR] SESSION(118114,ffff4effd1e0,python):2021-09-11-16:48:31.524.162 [mindspore/ccsrc/backend/session/ascend_session.cc:1857] ReportErrorMessage] Ascend error occurred, error message:
EZ9999: Inner Error!
        The device(3), core list[0-0], error code is:[FUNC:ProcessCoreErrorInfo][FILE:device_error_proc.cc][LINE:461]
        coreId( 0):        0x800000    [FUNC:ProcessCoreErrorInfo][FILE:device_error_proc.cc][LINE:472]
        Aicore kernel execute failed, device_id=0, stream_id=14, report_stream_id=17, task_id=13, fault kernel_name=add_13570952190744021808_0__kernel0, func_name=add_13570952190744021808_0__kernel0, program id=8, hash=14046644073217913723[FUNC:GetError][FILE:stream.cc][LINE:701]
        Stream synchronize failed, stream = 0xffff441cc970[FUNC:ReportError][FILE:logger.cc][LINE:566]
        rtStreamSynchronize execute failed, reason=[the model stream execute failed][FUNC:ReportError][FILE:error_message_manage.cc][LINE:566]

Traceback (most recent call last):
  File "try.py", line 34, in <module>
    output_1 = net(Tensor(x1), Tensor(x1))
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 391, in __call__
    out = self.compile_and_run(*inputs)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 671, in compile_and_run
    return _cell_graph_executor(self, *new_inputs, phase=self.phase)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 599, in __call__
    return self.run(obj, *args, phase=phase)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 627, in run
    return self._exec_pip(obj, *args, phase=phase_real)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 77, in wrapper
    results = fn(*arg, **kwargs)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 610, in _exec_pip
    return self._graph_executor(args_list, phase)
RuntimeError: mindspore/ccsrc/backend/session/ascend_session.cc:1456 Execute] run task error!
```

AICORE算子执行失败，可能是数据输入不匹配、访问越界、计算溢出等问题，也有可能是算子本身代码问题。针对此类问题用户可以通过日志和dump数据先自行排查，构造单算子用例进行定位，若无法定位出问题可以到[MindSpore社区](https://gitee.com/mindspore)提交issue获取帮助。

## AICPU算子执行问题

AICPU算子问题错误以`E3`开头。

### E39999: AICPU算子执行失败

一般AICPU算子执行失败会报`E39999`错误，同时MindSpore侧会有`Call rt api rtStreamSynchronize failed`的报错日志，根据错误码日志可能明确执行失败的算子，如下述报错场景：

```c++
[ERROR] RUNTIME(150840,python):2021-09-14-09:26:01.014.355 [engine.cc:914]150900 ReportExceptProc:Task exception! device_id=0, stream_id=117, task_id=23, type=13, retCode=0x91.
[ERROR] RUNTIME(150840,python):2021-09-14-09:26:01.014.485 [task.cc:600]150900 PrintAicpuErrorInfo:Aicpu kernel execute failed, device_id=0, stream_id=116, task_id=2, fault so_name=libaicpu_kernels.so, fault kernel_name=GetNext, fault op_name=, extend_info=.
[ERROR] RUNTIME(150840,python):2021-09-14-09:26:01.014.498 [task.cc:2292]150900 ReportErrorInfo:model execute error, retCode=0x91, [the model stream execute failed].
[ERROR] RUNTIME(150840,python):2021-09-14-09:26:01.014.506 [task.cc:2269]150900 PrintErrorInfo:model execute task failed, device_id=0, model stream_id=117, model task_id=23, model_id=1, first_task_id=65535
[ERROR] RUNTIME(150840,python):2021-09-14-09:26:01.014.548 [stream.cc:693]150900 GetError:Stream Synchronize failed, stream_id=117 retCode=0x91, [the model stream execute failed].
[ERROR] RUNTIME(150840,python):2021-09-14-09:26:01.014.594 [logger.cc:285]150900 StreamSynchronize:Stream synchronize failed, stream = 0xffff1c182e70
[ERROR] RUNTIME(150840,python):2021-09-14-09:26:01.014.622 [api_c.cc:581]150900 rtStreamSynchronize:ErrCode=507011, desc=[the model stream execute failed], InnerCode=0x7150050
[ERROR] RUNTIME(150840,python):2021-09-14-09:26:01.014.633 [error_message_manage.cc:41]150900 ReportFuncErrorReason:rtStreamSynchronize execute failed, reason=[the model stream execute failed]
[EXCEPTION] GE(150840,ffff2effd1e0,python):2021-09-14-09:26:01.014.680 [mindspore/ccsrc/runtime/device/ascend/ge_runtime/runtime_model.cc:233] Run] Call rt api rtStreamSynchronize failed, ret: 507011
[WARNING] DEVICE(150840,ffff2effd1e0,python):2021-09-14-09:26:01.015.138 [mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc:668] GetDumpPath] MS_OM_PATH is null, so dump to process local path, as ./rank_id/node_dump/...
[ERROR] DEVICE(150840,ffff2effd1e0,python):2021-09-14-09:26:01.015.177 [mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc:685] DumpTaskExceptionInfo] Task fail infos task_id: 2, stream_id: 116, tid: 150900, device_id: 0, retcode: 507011 ( model execute failed)
[ERROR] DEVICE(150840,ffff2effd1e0,python):2021-09-14-09:26:01.015.357 [mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc:694] DumpTaskExceptionInfo] Dump node (Default/GetNext-op1) task error input/output data to: ./rank_0/node_dump trace:
In file /home/jenkins/solution_test/remaining/common/features/dataset_base.py(1023)/        getnext_input, _ = self.get_next()/

[EXCEPTION] SESSION(150840,ffff2effd1e0,python):2021-09-14-09:26:01.060.662 [mindspore/ccsrc/backend/session/ascend_session.cc:1456] Execute] run task error!
[ERROR] SESSION(150840,ffff2effd1e0,python):2021-09-14-09:26:01.060.809 [mindspore/ccsrc/backend/session/ascend_session.cc:1857] ReportErrorMessage] Ascend error occurred, error message:
E39999: Inner Error!
        Aicpu kernel execute failed, device_id=0, stream_id=116, task_id=2, fault so_name=libaicpu_kernels.so, fault kernel_name=GetNext, fault op_name=, extend_info=[FUNC:GetError][FILE:stream.cc][LINE:701]
        Stream synchronize failed, stream = 0xffff1c182e70[FUNC:ReportError][FILE:logger.cc][LINE:566]
        rtStreamSynchronize execute failed, reason=[the model stream execute failed][FUNC:ReportError][FILE:error_message_manage.cc][LINE:566]

INFO 2021-09-14 09:26:01 - root - test_ms_cifar10_tdt_consume_beyond_produce_more_RDR.py:test_run:50 - when dataset batch num is less than train loop, error msg is mindspore/ccsrc/backend/session/ascend_session.cc:1456 Execute] run task error!
```

AICPU算子执行失败，可能是数据输入不匹配、访问越界、AICPU线程挂死等问题，也有可能是算子本身代码问题。针对此类问题用户可以通过日志和dump数据先自行排查，构造单算子用例进行定位，若无法定位出问题可以到[MindSpore社区](https://gitee.com/mindspore)提交issue获取帮助。

## runtime常见问题

Runtime模块对上承接MindSpore、ACL、GE、HCCL的调用，对下通过Driver模块对NPU上的各个模块进行调度，Runtime模块的错误码以`EE`开头。

### EE9999: HBM显存分配失败

当框架申请的HBM显存超过Device剩余显存时，就会报`halMemAlloc failed`错误，如下述错误场景所示：

```c++
[EXCEPTION] DEVICE(170414,fffe397fa1e0,python):2021-09-13-15:29:07.465.388 [mindspore/ccsrc/runtime/device/ascend/ascend_memory_manager.cc:62] MallocDeviceMemory] Malloc device memory failed, size[32212254720], ret[207001], Device 6 may be other processes occupying this card, check as: ps -ef|grep python
[ERROR] DEVICE(170414,fffe397fa1e0,python):2021-09-13-15:29:07.466.049 [mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc:375] Init] Ascend error occurred, error message:
EE9999: Inner Error!
        [driver interface] halMemAlloc failed: device_id=6, size=32212254720, type=2, env_type=3, drvRetCode=6![FUNC:ReportError][FILE:npu_driver.cc][LINE:568]
        [driver interface] halMemAlloc failed: size=32212254720, deviceId=6, type=2, env_type=3, drvRetCode=6![FUNC:ReportError][FILE:npu_driver.cc][LINE:568]
        DevMemAlloc huge page failed: deviceId=6, type=2, size=32212254720, retCode=117571606![FUNC:ReportError][FILE:npu_driver.cc][LINE:566]
        Device malloc failed, size=32212254720, type=2.[FUNC:ReportError][FILE:logger.cc][LINE:566]
        rtMalloc execute failed, reason=[driver error:out of memory][FUNC:ReportError][FILE:error_message_manage.cc][LINE:566]

Traceback (most recent call last):
  File "train.py", line 298, in <module>
    train_net()
  File "/home/jenkins/solution_test/remaining/test_scripts/fmea_multi_task/scripts/train/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
    run_func(*args, **kwargs)
  File "train.py", line 292, in train_net
    sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 715, in train
    sink_size=sink_size)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 500, in _train
    self._train_dataset_sink_process(epoch, train_dataset, list_callback, cb_params, sink_size)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 553, in _train_dataset_sink_process
    dataset_helper=dataset_helper)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 345, in _exec_preprocess
    dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/dataset_helper.py", line 245, in __init__
    self.iter = iterclass(dataset, sink_size, epoch_num)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/dataset_helper.py", line 393, in __init__
    super().__init__(dataset, sink_size, epoch_num)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/dataset_helper.py", line 300, in __init__
    create_data_info_queue=create_data_info_queue)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/_utils.py", line 73, in _exec_datagraph
    phase=phase)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 470, in init_dataset
    phase=phase):
RuntimeError: mindspore/ccsrc/runtime/device/ascend/ascend_memory_manager.cc:62 MallocDeviceMemory] Malloc device memory failed, size[32212254720], ret[207001], Device 6 may be other processes occupying this card, check as: ps -ef|grep python
```

遇到此类报错，可以先排查跑程序的卡是否已经被其他程序占用。目前MindSpore在Ascend环境上同一Device只支持同时跑一个程序，在910训练服务器上执行程序时会一次性申请32212254720KB（即30GB）的显存，故若报错信息中显示申请失败的显存大小为32212254720，则很有可能是该卡已经被其他程序占用，导致新程序申请显存失败。遇到这个问题只需确认卡未被其他程序占用后重新启动程序即可。

若报错信息中显示申请失败的显存大小不为32212254720，而是其他任意数字，则可能是网络模型太大，超过了Device的显存（910服务器为32GB），可以考虑改小batchsize、对网络模型进行优化或者使用模型并行等手段来作训练。

### EE8888: device id设置错误

用户可以通过环境变量DEVICE_ID或者在context中设置device_id来指定自己的程序跑在哪张卡上，如果device id设置不合理，则有可能会报`EE8888`错误，如下述错误场景，服务器中一共只有8张卡，可供选择的device id范围为[0, 8)，而用户错误设置了device_id=8：

```c++
[ERROR] DEVICE(127804,ffff964f1480,python):2022-05-28-15:32:37.021.254 [mindspore/ccsrc/runtime/device/context_extends.cc:103] OpenTsd] Ascend error occurred, error message:
EE8888: Inner Error!
        Set device failed, invalid device, current device=8, valid device range is [0, 8)[FUNC:SetDevice][FILE:api_error.cc][LINE:822]
        rtSetDevice execute failed, reason=[error device id][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:45]

[CRITICAL] DEVICE(127804,ffff964f1480,python):2022-05-28-15:32:37.021.316 [mindspore/ccsrc/runtime/device/context_extends.cc:105] OpenTsd] Device 8 call rtSetDevice failed, ret[107001]
Traceback (most recent call last):
  File "/home/jenkins/models/official/cv/lenet/scripts/../train.py", line 68, in <module>
    train_lenet()
  File "/home/jenkins/models/official/cv/lenet/src/model_utils/moxing_adapter.py", line 108, in wrapped_func
    run_func(*args, **kwargs)
  File "/home/jenkins/models/official/cv/lenet/scripts/../train.py", line 48, in train_lenet
    network = LeNet5(config.num_classes)
  File "/home/jenkins/models/official/cv/lenet/src/lenet.py", line 35, in __init__
    super(LeNet5, self).__init__()
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 117, in __init__
    init_pipeline()
RuntimeError: mindspore/ccsrc/runtime/device/context_extends.cc:105 OpenTsd] Device 8 call rtSetDevice failed, ret[107001]
```

遇到此类问题，根据错误日志排查device id的设置即可。

## HCCL&HCCP常见问题

HCCL(Huawei Collective Communication Library)是华为集合通信库，提供了深度学习训练场景中服务器间高性能集合通信功能，其通信过程分为如下三个阶段：

1. 通信初始化：获取必要的集合通信参数配置并初始化网络设备。初始化阶段不涉及不同设备之前的交互。

2. 建立通信连接：建立socket连接并交换通信两端的通信参数和内存信息。建立通信连接阶段，HCCL会根据用户提供的集群信息结合网络拓扑与其他卡进行建链并交换用于通信的参数信息。如果在建链超时时间阈值（MindSpore默认设置600s，可通过环境变量HCCL_CONNECT_TIMEOUT配置）内未得到其他卡的及时响应，会上报建链超时错误并退出训练。

3. 执行通信操作：通过Notify同步设备执行状态，传递内存数据。通信操作执行阶段，HCCL会根据通信算法编排NOTIFY/SDMA等task并通过runtime下发给昇腾设备task调度器，设备根据编排信息调度并执行task。其中Notify类task用于卡间同步，Notify wait会阻塞task流执行直到对应的Notify record到达，以确保后续的通信操作执行时彼此的内存处于ready状态。

HCCL&HCCP的错误码以`EI`和`EJ`开头，在整个通信过程中，集群中出现的单卡问题、通信链路问题均可能会导致集群出现大量的超时错误，因此在定位集群通信问题时需要收集整个集群的日志信息，锁定问题出现的位置。

### socket建链超时

当socker建链超时时，MindSpore日志会出现`Distribute Task Failed`的报错，若同时打印CANN的日志发现HCCL模块的日志中出现`get socket timeout`的报错，则说明集群出现socket建链超时错误，如下述日志所示：

```c++
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.375 [exchanger_network.cc:232][130250][138166][Wait][AllClientSocketEstab]errNo[0x000000000500000b] client : device[1] rank[1] get socket timeout, total[4] remain[3]
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.435 [exchanger_network.cc:378][130250][138166]call trace: ret -> 4
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.446 [exchanger_network.cc:104][130250][138166][ExchangerNetwork][Init]rank[1] device[1] wait all socket establish 120 second failed
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.509 [exchanger_network.cc:760][130250][138166]   _____________________________________________________
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.519 [exchanger_network.cc:761][130250][138166]   |device[1] userrank[1] exchanger Status: run_step[1]|
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.526 [exchanger_network.cc:762][130250][138166]   |  dest_dev  |  userrank  |    Role    | connStatus |
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.533 [exchanger_network.cc:763][130250][138166]   |------------|------------|------------|------------|
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.543 [exchanger_network.cc:782][130250][138166]   |         0  |         0  |   client   |     NO     |
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.550 [exchanger_network.cc:782][130250][138166]   |         1  |         1  |     NA     |     NA     |
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.557 [exchanger_network.cc:782][130250][138166]   |         2  |         2  |   server   |     NO     |
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.588 [exchanger_network.cc:782][130250][138166]   |         3  |         3  |   client   |     YES     |
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.594 [exchanger_network.cc:782][130250][138166]   |         4  |         4  |   server   |     NO     |
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.601 [exchanger_network.cc:782][130250][138166]   |         5  |         5  |   client   |     NO     |
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.607 [exchanger_network.cc:782][130250][138166]   |         6  |         6  |   server   |     NO     |
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.612 [exchanger_network.cc:782][130250][138166]   |         7  |         7  |   client   |     NO     |
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.619 [exchanger_network.cc:787][130250][138166]   ___________________________________________________________________
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.625 [exchanger_network.cc:788][130250][138166]the connection failure between this device and the target device may be due to the following reasons:
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.631 [exchanger_network.cc:789][130250][138166]1. the connection between this device and the target device is abnormal.
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.637 [exchanger_network.cc:790][130250][138166]2. an exception occurred at the target devices.
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.644 [exchanger_network.cc:792][130250][138166]3. the time difference between the execution of hcom on this device and the target device exceeds the timeout threshold. make sure this by keyworld [Entry-]
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.650 [exchanger_network.cc:794][130250][138166]4. the behavior of executing the calculation graph on this device and the target device is inconsistent.
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.787.658 [comm_factory.cc:1102][130250][138166][Get][ExchangerNetwork]exchanger init failed
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.830.177 [comm_factory.cc:244][130250][138166][Create][CommOuter]exchangerNetwork create failed
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.830.192 [hccl_impl.cc:2105][130250][138166][Create][OuterComm]errNo[0x0000000005000006] tag[HcomAllReduce_6629421139219749105_0], created commOuter fail. commOuter[0] is null
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.830.276 [hccl_impl.cc:1878][130250][130825][Create][CommByAlg]CreateInnerComm [0] or CreateOuterComm[6] failed. commType[2]
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.830.311 [hccl_impl.cc:1975][130250][130825]call trace: ret -> 4
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.830.331 [hccl_impl.cc:896][130250][130825][HcclImpl][AllReduce]errNo[0x0000000005000004]  tag[HcomAllReduce_6629421139219749105_0],all reduce create comm failed
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.830.340 [hccl_comm.cc:240][130250][130825]call trace: ret -> 4
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.830.351 [hcom.cc:251][130250][130825][AllReduce][Result]errNo[0x0000000005010004] hcclComm all reduce error, tag[HcomAllReduce_6629421139219749105_0],input_ptr[0x108a39a06400], output_ptr[0x108a02805200], count[22348800], data_type[4], op[0], stream[0xfffeb6a2ccd0]
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.830.363 [hcom_ops_kernel_info_store.cc:357][130250][130825]call trace: ret -> 4
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.830.386 [hcom_ops_kernel_info_store.cc:191][130250][130825]call trace: ret -> 4
[ERROR] HCCL(130250,python):2021-10-14-00:50:50.830.394 [hcom_ops_kernel_info_store.cc:854][130250][130825][Load][Task]errNo[0x0000000005010004] load task failed. (load op[HcomAllReduce] fail)
[EXCEPTION] GE(130250,fffe49ffb1e0,python):2021-10-14-00:50:50.830.457 [mindspore/ccsrc/runtime/device/ascend/ge_runtime/task/hccl_task.cc:99] Distribute] davinci_model : load task fail, return ret: 1343225860
[EXCEPTION] DEVICE(130250,fffe49ffb1e0,python):2021-10-14-00:50:58.410.094 [mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc:547] LoadTask] Distribute Task Failed, error: mindspore/ccsrc/runtime/device/ascend/ge_runtime/task/hccl_task.cc:99 Distribute] davinci_model : load task fail, return ret: 1343225860
```

如日志中所说，socket建链超时常见的的原因有：

1. 部分卡未执行到正确的建链阶段，在之前已出错；

2. 部分卡被某些耗时较长的任务阻塞，在超过600秒（可通过HCCL_CONNECT_TIMEOUT配置）后才执行到对应阶段；

3. 网络模型等原因导致某些卡间的通信算子数量或者排序不一致；

4. 节点间通信链路不通或者不稳定。

在收集了集群所有卡的INFO日志（包括CANN日志）后，可以按照以下步骤进行排查：

1. 检查所有卡的报错日志，若有卡未报socket建链超时错误，可以通过日志时间检查判断此卡是否存在业务进程报错退出、卡死或core宕机的情况导致集群socket建链超时，然后转单卡问题定位；

2. 若所有卡均上报socket建链超时错误，则检查各卡的错误日志中最早和最晚的时间差异是否超过超时阈值，若超过阈值请定位报错时间最晚的rank执行阻塞原因或者调整超时阈值（默认为600秒，通过环境变量HCCL_CONNECT_TIMEOUT设置）；

3. 检查集群中是否存在Device网口通信链路不通的情况，比较常见的原因：

    a. IP不在同一网段或子网掩码配置存在问题;

    b. IP冲突，集群中存在IP相同的两个rank;

    c. 各rank的TLS(安全增强)设置不一致。

### EI0002: notify wait超时

常见于执行阶段，HCCL算子的task会在指定集群的每个Device上执行，并通过notify进行状态同步，若任何一张卡或者通信链路在执行前/中发生异常，则会导致集群同步失败，剩余卡会出现notify wait超时，报`EI0002`错误，如下所示：

```c++
[ERROR] DEVICE(105395,fffdda7fc0f0,python):2022-11-29-10:20:42.200.963 [mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc:744] DumpTaskExceptionInfo] Task fail infos task_id: 10, stream_id: 27, tid: 105395, device_id: 1, retcode: 507011 ( model execute failed)
[WARNING] DEVICE(105395,fffd3d7fa0f0,python):2022-11-29-10:20:42.340.685 [mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_data_queue.cc:257] Push] Device queue thread had been interrupted by TdtHandle::DestroyHandle, you can ignore the above error: 'failed to send...'. In this scenario, the training ends first without using all epoch(s) data, and the data preprocessing is blocked by the data transmission channel on the device side. So we force the data transmission channel to stop.
[WARNING] MD(105395,fffd3d7fa0f0,python):2022-11-29-10:20:42.340.894 [mindspore/ccsrc/minddata/dataset/engine/datasetops/data_queue_op.cc:257] SendDataToAscend] Thread has already been terminated.
Traceback (most recent call last):
  File "train.py", line 377, in <module>
    train_net()
  File "/home/jenkins/ResNet/scripts/train_parallel1/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
    run_func(*args, **kwargs)
  File "train.py", line 370, in train_net
    sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 1052, in train
    initial_epoch=initial_epoch)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 98, in wrapper
    func(self, *args, **kwargs)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 614, in _train
    cb_params, sink_size, initial_epoch, valid_infos)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 692, in _train_dataset_sink_process
    outputs = train_network(*inputs)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 627, in __call__
    out = self.compile_and_run(*args)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 946, in compile_and_run
    return _cell_graph_executor(self, *new_inputs, phase=self.phase)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 1371, in __call__
    return self.run(obj, *args, phase=phase)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 1408, in run
    return self._exec_pip(obj, *args, phase=phase_real)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 98, in wrapper
    results = fn(*arg, **kwargs)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 1390, in _exec_pip
    return self._graph_executor(args, phase)
RuntimeError: Run task for graph:kernel_graph_1 error! The details refer to 'Ascend Error Message'.

----------------------------------------------------
- Ascend Error Message:
----------------------------------------------------
EI0002: The wait execution of the Notify register times out. Reason: The Notify register has not received the Notify record from remote rank [0].base information: [streamID:[27], taskID[10], taskType[Notify Wait], tag[HcomAllReduce_6629421139219749105_0].] task information: [notify id:[0x0000000100000000], stage:[ffffffff], remote rank:[0].
there are(is) 1 abnormal device(s):
        serverId[10.90.55.95], deviceId[0], Heartbeat Lost Occurred, Possible Reason: 1. Process has exited, 2. Network Disconnected
]
        Possible Cause: 1. An exception occurs during the execution on some NPUs in the cluster. As a result, collective communication operation failed.2. The execution speed on some NPU in the cluster is too slow to complete a communication operation within the timeout interval. (default 1800s, You can set the interval by using HCCL_EXEC_TIMEOUT.)3. The number of training samples of each NPU is inconsistent.4. Packet loss or other connectivity problems occur on the communication link.
        Solution: 1. If this error is reported on part of these ranks, check other ranks to see whether other errors have been reported earlier.2. If this error is reported for all ranks, check whether the error reporting time is consistent (the maximum difference must not exceed 1800s). If not, locate the cause or adjust the locate the cause or set the HCCL_EXEC_TIMEOUT environment variable to a larger value.3. Check whether the completion queue element (CQE) of the error exists in the plog(grep -rn 'error cqe'). If so, check the network connection status. (For details, see the TLS command and HCCN connectivity check examples.)4. Ensure that the number of training samples of each NPU is consistent.
        TraceBack (most recent call last):
        Notify wait execute failed, device_id=1, stream_id=27, task_id=10, flip_num=0, notify_id=0[FUNC:GetError][FILE:stream.cc][LINE:921]
        rtStreamSynchronize execute failed, reason=[the model stream execute failed][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:49]

(Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_graph_executor.cc:240 RunGraph
```

如日志中所说，notify wait超时常见的的原因有：

1. 部分卡未能成功执行到notify同步阶段，在之前已出错；

2. 部分卡被某些耗时较长的任务阻塞，在超过1800秒（可通过HCCL_EXEC_TIMEOUT配置）后才执行到对应阶段；

3. 网络模型等原因导致某些卡间的task执行序列不一致；

4. 节点间通信链路不稳定。

在收集了集群所有卡的INFO日志（包括CANN日志）后，可以按照以下步骤进行排查：

1. 检查所有卡的报错日志，若有卡未报notify wait超时错误，可以通过日志时间检查判断此卡是否存在业务进程报错退出、卡死或core宕机的情况导致集群notify wait超时，然后转单卡问题定位；

2. 若所有卡均上报notify wait超时错误，则检查各卡的错误日志中最早和最晚的时间差异是否超过超时阈值，若超过阈值请定位报错时间最晚的rank执行阻塞原因（如save checkpoint）或者调整超时阈值（默认为1800秒，通过环境变量HCCL_EXEC_TIMEOUT设置）；

3. 检查集群中是否存在Device网口通信链路不稳定的情况，排查所有卡的Device侧日志，若存在error cqe的打印且时间位于业务区间内，则请定位网络丢包的原因。

### EI0004: 非法ranktable配置

用户需要通过ranktable文件来配置分布式训练需要的多机多卡信息以供HCCL初始化，若相关ranktable配置非法，则会报`EI0004`错误，如下述报错场景，ranktable中重复设置了两个device_id为1，导致配置非法：

```c++
[WARNING] HCCL_ADPT(86172,ffffb4d8e480,python):2019-07-25-19:41:56.313.202 [mindspore/ccsrc/runtime/hccl_adapter/hccl_adapter.cc:58] GenHcclOptions] The environment variable DEPLOY_MODE is not set. Now set to default value 0
[CRITICAL] HCCL_ADPT(86172,ffffb4d8e480,python):2019-07-25-19:41:56.314.843 [mindspore/ccsrc/runtime/hccl_adapter/hccl_adapter.cc:391] InitKernelInfoStore] Init hccl graph adapter failed.
[CRITICAL] DEVICE(86172,ffffb4d8e480,python):2019-07-25-19:41:56.315.092 [mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc:357] Init] Ascend error occurred, error message: EI0004: Invalid ranktable, with rank_id [0] and local device_id [0]. Check that ranktable [{"server_count":"1","server_list":[{"device":[{"device_id":"0","device_ip":"192.168.100.101","rank_id":"0"},{"device_id":"1","device_ip":"192.168.101.101","rank_id":"1"},{"device_id":"1","device_ip":"192.168.102.101","rank_id":"2"},{"device_id":"3","device_ip":"192.168.103.101","rank_id":"3"},{"device_id":"4","device_ip":"192.168.100.102","rank_id":"4"},{"device_id":"5","device_ip":"192.168.101.102","rank_id":"5"},{"device_id":"6","device_ip":"192.168.102.102","rank_id":"6"},{"device_id":"7","device_ip":"192.168.103.102","rank_id":"7"}],"host_nic_ip":"reserve","server_id":"10.174.229.253"}],"status":"completed","version":"1.0"}] is valid and the environment setup matches the ranktable.

First error scene API: mindspore/ccsrc/runtime/hccl_adapter/hccl_adapter.cc:391 InitKernelInfoStore] Init hccl graph adapter failed.
[WARNING] DEBUG(86172,ffffb4d8e480,python):2019-07-25-19:41:56.316.394 [mindspore/ccsrc/debug/rdr/recorder_manager.cc:108] TriggerAll] There is no recorder to export.

Traceback (most recent call last):
  File "train.py", line 392, in <module>
    train_net()
  File "/home/jenkins/models/official/cv/resnet/scripts/train_parallel0/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
    run_func(*args, **kwargs)
  File "train.py", line 315, in train_net
    set_parameter()
  File "train.py", line 162, in set_parameter
    init()
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/communication/management.py", line 142, in init
    init_hccl()
RuntimeError: mindspore/ccsrc/runtime/device/ascend/ascend_kernel_runtime.cc:357 Init] Ascend error occurred, error message: EI0004: Invalid ranktable, with rank_id [0] and local device_id [0]. Check that ranktable [{"server_count":"1","server_list":[{"device":[{"device_id":"0","device_ip":"192.168.100.101","rank_id":"0"},{"device_id":"1","device_ip":"192.168.101.101","rank_id":"1"},{"device_id":"1","device_ip":"192.168.102.101","rank_id":"2"},{"device_id":"3","device_ip":"192.168.103.101","rank_id":"3"},{"device_id":"4","device_ip":"192.168.100.102","rank_id":"4"},{"device_id":"5","device_ip":"192.168.101.102","rank_id":"5"},{"device_id":"6","device_ip":"192.168.102.102","rank_id":"6"},{"device_id":"7","device_ip":"192.168.103.102","rank_id":"7"}],"host_nic_ip":"reserve","server_id":"10.174.229.253"}],"status":"completed","version":"1.0"}] is valid and the environment setup matches the ranktable.

First error scene API: mindspore/ccsrc/runtime/hccl_adapter/hccl_adapter.cc:391 InitKernelInfoStore] Init hccl graph adapter failed.
```

遇到此类报错，根据报错日志对ranktable配置文件进行修正即可。

### EI0005: 卡间通信参数不一致

当卡间通信参数不一致，比如卡间用于AllReduce的输入shape大小不一致时，就会报`EI0005`错误，如下述报错场景，有个名为count的parameter在卡间通信时大小不一致：

```c++
[CRITICAL] GE(98494,ffff9c51e480,python):2022-04-28-10:56:52.072.619 [mindspore/ccsrc/plugin/device/ascend/hal/device/ge_runtime/task/hccl_task.cc:100] Distribute] davinci_model : load task fail, return ret: 1343225860
[CRITICAL] DEVICE(98494,ffff9c51e480,python):2022-04-28-10:56:52.072.964 [mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc:567] LoadTask] Distribute Task Failed,
error msg: mindspore/ccsrc/plugin/device/ascend/hal/device/ge_runtime/task/hccl_task.cc:100 Distribute] davinci_model : load task fail, return ret: 1343225860
[ERROR] DEVICE(98494,ffff9c51e480,python):2022-04-28-10:56:52.073.054 [mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_device_context.cc:660] ReportErrorMessage] Ascend error occurred, error message:
EI0005: The arguments for collective communication are inconsistent between ranks: tag [HcomAllReduce_6629421139219749105_0], parameter [count], local [9556480], remote [9555712]

[CRITICAL] DEVICE(98494,ffff9c51e480,python):2022-04-28-10:56:52.073.078 [mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_device_context.cc:422] PreprocessBeforeRunGraph] Preprocess failed before run graph 0,
error msg: mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc:567 LoadTask] Distribute Task Failed,
error msg: mindspore/ccsrc/plugin/device/ascend/hal/device/ge_runtime/task/hccl_task.cc:100 Distribute] davinci_model : load task fail, return ret: 1343225860
dataset length:  848
data pre-process time is 0.04648423194885254

Traceback (most recent call last):
  File "e2e_feed_dev.py", line 294, in <module>
    run()
  File "e2e_feed_dev.py", line 277, in run
    label_indices, label_values)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 586, in __call__
    out = self.compile_and_run(*args)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 964, in compile_and_run
    self.compile(*inputs)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 937, in compile
    _cell_graph_executor.compile(self, *inputs, phase=self.phase, auto_parallel_mode=self._auto_parallel_mode)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 1006, in compile
    result = self._graph_executor.compile(obj, args_list, phase, self._use_vm_mode())
RuntimeError: mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_device_context.cc:422 PreprocessBeforeRunGraph] Preprocess failed before run graph 0,
error msg: mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc:567 LoadTask] Distribute Task Failed,
error msg: mindspore/ccsrc/plugin/device/ascend/hal/device/ge_runtime/task/hccl_task.cc:100 Distribute] davinci_model : load task fail, return ret: 1343225860
```

遇到此类报错，可通过日志和IR图确认出错的通信参数，并在网络脚本中予以修正。

### EJ0001: HCCP初始化失败

HCCP进程负责实现通信功能，HCCL可以调用HCCP的接口进行通信。HCCP初始化失败会报`EJ0001`错误，比如以下场景，当上一个八卡训练任务还未结束时就在同一服务器启动新的八卡训练任务就会导致初始化失败，需要等之前的八卡训练任务结束后才能启动新的八卡训练任务。

```c++
[WARNING] HCCL_ADPT(57647,ffff89ae7010,python):2022-07-28-18:56:01.822.555 [mindspore/ccsrc/plugin/device/ascend/hal/hccl_adapter/hccl_adapter.cc:47] GenHcclOptions] The environment variable DEPLOY_MODE is not set. Now set to default value 0

Traceback (most recent call last):
  File "train.py", line 380, in <module>
    train_net()
  File "/home/jenkins/models/official/cv/resnet/scripts2/train_parallel0/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
    run_func(*args, **kwargs)
  File "train.py", line 307, in train_net
    set_parameter()
  File "train.py", line 154, in set_parameter
    init()
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/communication/management.py", line 146, in init
    init_hccl()
RuntimeError: Ascend error occurred, error message: EJ0001: Failed to initialize the HCCP process. Reason: The last training process is running.
        Solution: Wait for 10s after killing the last training process and try again.
        tsd client wait response fail, device response code[1]. unknown device error.[FUNC:WaitRsp][FILE:process_mode_manager.cpp][LINE:229]

First error scene API: Init hccl graph adapter failed.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc:363 Init
mindspore/ccsrc/plugin/device/ascend/hal/hccl_adapter/hccl_adapter.cc:408 InitKernelInfoStore
```

遇到此类报错，根据日志的初始化错误原因及解决方案进行解决即可。

## profiling常见问题

profiling问题错误码以`EK`开头。

### EK0001: 非法参数问题

当CANN的profiling模块接口遇到非法参数时会报`EK0001`错误。对用户来说，最常见的非法参数错误就是profiling的路径设置不对，比如下述报错：

```c++
[ERROR] PROFILER(138694,ffffaa6c8480,python):2022-01-10-14:19:56.741.053 [mindspore/ccsrc/profiler/device/ascend/ascend_profiling.cc:51] ReportErrorMessage] Ascend error occurred, error message:
EK0001: Path [/ms_test/ci/user_scene/profiler_chinese_中文/resnet/scripts/train/data/profiler] for [profilerResultPath] is invalid or does not exist. The Path name can only contain A-Za-z0-9-_.

[CRITICAL] PROFILER(138694,ffffaa6c8480,python):2022-01-10-14:19:56.741.123 [mindspore/ccsrc/profiler/device/ascend/ascend_profiling.cc:79] InitProfiling] Failed to call aclprofInit function.

Traceback (most recent call last):
  File "train.py", line 387, in <module>
    train_net()
  File "/ms_test/ci/user_scene/profiler_chinese_中文/resnet/scripts/train/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
    run_func(*args, **kwargs)
  File "train.py", line 325, in train_net
    profiler = Profiler()
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/profiler/profiling.py", line 195, in __init__
    self._ascend_profiler.init(self._output_path, int(self._dev_id), profiling_options)
RuntimeError: mindspore/ccsrc/profiler/device/ascend/ascend_profiling.cc:79 InitProfiling] Failed to call aclprofInit function.
```

从ERROR日志的CANN报错信息中可以获知，profiling的路径字符只能包含`A-Za-z0-9-_`，而上述报错中的profiling路径中含有`中文`两字，导致路径非法报错，从而导致profiling初始化失败。遇到此类问题，根据报错信息改正profiling路径或者其他参数即可。
