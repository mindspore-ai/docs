# CANN Common Error Analysis

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/tutorials/source_en/debug/error_analysis/cann_error_cases.md)&nbsp;&nbsp;

This article focuses on the handling of common CANN errors by users. When encountering CANN errors, MindSpore logs may not be sufficient to analyze the related errors. You can print CANN logs to better analyze the errors by setting the following two environment variables:

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1        # CANN log level, 0 for debug, 1 for info, 2 for warning, 3 for error
export ASCEND_SLOG_PRINT_TO_STDOUT=1    # Configure to enable log displaying
```

For more CANN errors and log settings, please refer to the `Troubleshooting` document section of [Ascend Community](https://www.hiascend.com/en/document).

## AICORE Operator Compilation Problem

AICORE operator compilation errors will start with `E5`~`EB` according to different modules, where some checksum errors occur in the `E5`~`E8` error codes AICORE operator compilation process, and the `E9`~`EB` error code is the error reported by TBE before and after compilation. Generally speaking an error report means that the operator specification TBE is not yet supported, and specific information can be obtained from the error report log. Some specific AICORE operator compilation failure problems can be seen below:

### E80000: StridedSliceGradD Illegal Input Value

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
In file /home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/ops/_grad_experimental/grad_array_ops.py(700)/        dx = input_grad(dout, x_shape, begin, end, strides)/
Corresponding forward node candidate:
- In file /home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/ops/composite/multitype_ops/_compile_utils.py(306)/        return P.StridedSlice(begin_mask, end_mask, 0, 0, 0)(data, begin_strides, end_strides, step_strides)/
  In file /home/jenkins/models/official/cv/lenet/src/lenet.py(61)/        y = x[0::2] #Splitting operation, x.shape=(32,10) y.shape=(32), leading to dimensionality reduction/
  In file /home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/amp.py(126)/            out = self._backbone(data)/
  In file /home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/wrap/loss_scale.py(332)/        loss = self.network(*inputs)/
  In file /home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/dataset_helper.py(95)/        return self.network(*outputs)/
```

From the error message of the above error code, we can see that the first data in the parameter strides of StridedSliceGradD operator is expected to be 1, but the actual one obtained is 2, so the error is reported. Users can confirm the information of the wrong operator and correct the strides parameter according to the IR diagram.

### E80012: ReduceSum Operator Input Dimension Is Too High

```c++
RuntimeError: ({'errCode': 'E80012', 'op_name': 'reduce_sum_d', 'param_name': 'x', 'min_value': 0, 'max_value': 8, 'real_value': 10}, 'In op, the num of dimensions of input/output[x] should be in the range of [0, 8], but actually is [10].')
```

As you can see from the error message in the error code above, the input and output of the ReduceSum operator only supports 8 dimensions of data, but it actually encounters 10 dimensions of data, hence the error occurs. The user can circumvent this error by modifying the network script to avoid inputting 10-dimensional data to ReduceSum.

### E80029: Assign Operator Shape Inconsistency

```c++
RuntimeError: ({'errCode': 'E80029', 'op_name': 'assign', 'param_name1': 'ref', 'param_name2': 'value', 'error_detail': 'Shape of ref and value should be same'}, 'In op[assign], the shape of inputs[ref][value] are invalid, [Shape of ref and value should be same].')
```

The logic of the Assign operator is to use the second input (i.e. value) to assign a value to the parameter (i.e. ref) of the first input. From the error message of the above error code, we can see that the two inputs are expected to have the same shape, but they do not, so the error is reported. The user can identify the operator in error and correct the input of the Assign operator based on the IR graph.

### EB0000: Transpose Specifications Are Not Supported

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

input_args: {"SocInfo": {"autoTilingMode": "NO_TUNE", "coreNum": "", "coreType": "", "l1Fusion": "false", "l2Fusion": "false", "l2Mode": "2", "op_debug_level": "", "op_impl_mode": "", "op_impl_mode_list": [], "socVersion": "Ascend910"}, "impl_path": "", "op_info": {"Type": "Transpose", "attr_desc": [[0, 2, 1, 3]], "attrs": [{"name": "perm", "valid": true, "value": [0, 2, 1, 3]}], "full_name": "Default/encode_image-EncodeImage/visual-ViT/body-Transformer/layers-SequentialCell/0-SequentialCell/0-ResidualCell/cell-SequentialCell/1-AttentionWithMask/Transpose-op662", "gen_model": "single", "graph_id": 0, "inputs": [[{"addr_type": 0, "dtype": "float16", "format": "NCHW", "name": "x_0", "ori_format": "NCHW", "ori_shape": [256, 16, 257, 64], "param_type": "required", "range": [[256, 256], [16, 16], [257, 257], [64, 64]], "shape": [256, 16, 257, 64], "valid": true}]], "is_dynamic_shape": false, "kernel_name": "Transpose_4779120815397556904_6", "module_name": "impl.transpose_d", "name": "transpose_d", "op_tune_list": "ALL", "op_tune_switch": "on", "outputs": [[{"addr_type": 0, "dtype": "float16", "format": "NCHW", "name": "y", "ori_format": "NCHW", "ori_shape": [256, 257, 16, 64], "param_type": "required", "range": [[256, 256], [257, 257], [16, 16], [64, 64]], "shape": [256, 257, 16, 64], "valid": true}]], "pass_list": "ALL", "py_module_path": "/usr/local/Ascend/opp/op_impl/built-in/ai_core/tbe", "rl_tune_list": "ALL", "rl_tune_switch": "on", "socVersion": "Ascend910"}, "platform": "TBE", "reset_op_info": [{"type": "clear_vector", "bin_path": "./kernel_meta/vector_random_buff.o", "kernel_name": "vector_random_buff"}, {"type": "clear_cube", "bin_path": "./kernel_meta/cube_random_buff.o", "kernel_name": "cube_random_buff"}]} trace:
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

From the error message reported in the above error code, we can see that the specification of this transpose operator is not supported and its data exceeds the limit when doing memory copy instructions. For such problems, users can submit issues to [MindSpore Community](https://gitee.com/mindspore) for help.

## AICORE Operator Execution Problem

### EZ9999: AICORE Operator Execution Failure

Generally AICORE operator execution failure will report `EZ9999` error, while MindSpore side will have `Call rt api rtStreamSynchronize failed` error log. According to the error code log, it may be possible to specify the operator that failed to execute, such as the following error reporting scenario where the execution of the Add operator failed:

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

AICORE operator execution failure may be due to data input mismatch, access out-of-bounds, computation overflow, or it may be a code problem with the operator itself. For this kind of problem, users can use the logs and dump data to first check by themselves and construct a single-calculus use case to locate. If you cannot locate the problem, you can submit an issue to the [MindSpore Community](https://gitee.com/mindspore) for help.

## AICPU Operator Execution Problem

The AICPU operator problem error starts with `E3`.

### E39999: AICPU Operator Execution Failure

Generally the AICPU operator will report `E39999` error during operator execution failure, while there will be `Call rt api rtStreamSynchronize failed` error log on MindSpore side. According to the error code log, it is possible to specify the operator that failed to execute, such as the following error reporting scenario:

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

The AICPU operator execution failure may be due to data input mismatch, access out-of-bounds, AICPU thread hang, or caused by the operator itself. For this kind of problem, users can use the logs and dump data to first check by themselves and construct a single-calculus use case to locate. If you cannot locate the problem, you can submit an issue to the [MindSpore Community](https://gitee.com/mindspore) for help.

## runtime FAQ

The Runtime module takes over the calls of MindSpore, ACL, GE, HCCL and schedules each module on the NPU through the Driver module. The error code of the Runtime module starts with `EE`.

### EE9999: On-Chip Video Memory Allocation Failure

When the On-Chip memory requested by the framework exceeds the remaining memory of the Device, a `halMemAlloc failed` error is reported, as shown in the following error scenario:

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

When encounter such an error, you can first check whether the card running the program is already occupied by another program. At present, MindSpore only supports one program running on the same Device in Ascend environment, and the program will apply for 32212254720KB (i.e. 30GB) of video memory at one time when it is executed on the 910 training server, so if the error message shows that the size of video memory in the failed application is 32212254720, it is likely that the card is already occupied by other programs, resulting in the failed application of video memory for the new program. When you encounter this problem just make sure the card is not occupied by another program and restart the program.If the error message shows that the size of video memory in the failed application is not 32212254720, but any other number, the network model may be too large and exceed the Device memory (32GB for 910 servers), consider changing the batchsize, optimizing the network model or using model parallelism for training.

In addition, currently MindSpore will verify the remaining Device memory of the device during program initialization. If the remaining Device memory is less than half of the total, the following error will be reported to indicate that the device is occupied:

```c++
[CRITICAL] DEVICE(164104,ffff841795d0,python):2022-12-01-03:58:52.033.238 [mindspore/ccsrc/runtime/device/kernel_runtime.cc:124] LockRuntime] The pointer[stream] is null.
[ERROR] DEVICE(164104,ffff841795d0,python):2022-12-01-03:58:52.033.355 [mindspore/ccsrc/runtime/device/kernel_runtime_manager.cc:138] WaitTaskFinishOnDevice] SyncStream failed, exception:The pointer[stream] is null.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/runtime/device/kernel_runtime.cc:124 LockRuntime

Traceback (most recent call last):
  File "train.py", line 377, in <module>
    train_net()
  File "/home/jenkins/workspace/TDT_deployment/solution_test/remaining/test_scripts/mindspore/reliability/fmea/business/process/multitask/test_ms_fmea_multi_task_1p_1p_0001_2_GRAPH_MODE/scripts/train/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
    run_func(*args, **kwargs)
  File "train.py", line 370, in train_net
    sink_size=dataset.get_dataset_size(), dataset_sink_mode=dataset_sink_mode)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 1052, in train
    initial_epoch=initial_epoch)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 98, in wrapper
    func(self, *args, **kwargs)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 614, in _train
    cb_params, sink_size, initial_epoch, valid_infos)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 673, in _train_dataset_sink_process
    dataset_helper=dataset_helper)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/model.py", line 427, in _exec_preprocess
    dataset_helper = DatasetHelper(dataset, dataset_sink_mode, sink_size, epoch_num)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/dataset_helper.py", line 335, in __init__
    self.iter = iterclass(dataset, sink_size, epoch_num)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/dataset_helper.py", line 530, in __init__
    super().__init__(dataset, sink_size, epoch_num)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/dataset_helper.py", line 429, in __init__
    create_data_info_queue=create_data_info_queue)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/train/_utils.py", line 74, in _exec_datagraph
    phase=phase)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 1264, in init_dataset
    need_run=need_run):
RuntimeError: Ascend kernel runtime initialization failed. The details refer to 'Ascend Error Message'.

----------------------------------------------------
- Framework Error Message:
----------------------------------------------------
Malloc device memory failed, free memory size is less than half of total memory size.Device 0 Device MOC total size:34359738368 Device MOC free size:2140602368 may be other processes occupying this card, check as: ps -ef|grep python
```

### EE9999: Runtime Task Execution Failure

The operator or task of runtime is generally related to the control flow, such as event wait, send, recv, etc. When the task of runtime fails, it will report an `EE9999 Task execute failed` error, as below:

```c++
[CRITICAL] DEVICE(160186,fffd5affd0f0,python):2023-01-10-09:15:46.798.038 [mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_data_queue.cc:291] ParseType] Got unsupported acl datatype: -1
[ERROR] MD(160186,fffd5affd0f0,python):2023-01-10-09:15:46.798.688 [mindspore/ccsrc/minddata/dataset/util/task.cc:75] operator()] Unexpected error. Got unsupported acl datatype: -1

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_data_queue.cc:291 ParseType

Line of code : 74
File         : mindspore/ccsrc/minddata/dataset/util/task.cc

[ERROR] MD(160186,fffd5affd0f0,python):2023-01-10-09:15:46.798.741 [mindspore/ccsrc/minddata/dataset/util/task_manager.cc:222] InterruptMaster] Task is terminated with err msg (more details are in info level logs): Unexpected error. Got unsupported acl datatype: -1

INFO 2023-01-10 09:15:46 - train - dataset_func.py:op_network_with_create_dict_iterator:302 - Iter Num : 0
INFO 2023-01-10 09:15:46 - train - dataset_func.py:op_network_with_create_dict_iterator:302 - Iter Num : 0
INFO 2023-01-10 09:15:46 - train - dataset_func.py:op_network_with_create_dict_iterator:302 - Iter Num : 0
INFO 2023-01-10 09:15:46 - train - dataset_func.py:op_network_with_create_dict_iterator:302 - Iter Num : 0
INFO 2023-01-10 09:15:46 - train - dataset_func.py:op_network_with_create_dict_iterator:302 - Iter Num : 0
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.322.063 [engine.cc:1263]163618 ReportExceptProc:Task exception! device_id=0, stream_id=17, task_id=43, type=13, retCode=0x91, [the model stream execute failed].
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.323.519 [task.cc:92]163618 PrintErrorInfo:Task execute failed, base info: device_id=0, stream_id=2, task_id=3, flip_num=0, task_type=3.
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.323.542 [task.cc:3239]163618 ReportErrorInfo:model execute error, retCode=0x91, [the model stream execute failed].
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.323.553 [task.cc:3210]163618 PrintErrorInfo:model execute task failed, device_id=0, model stream_id=17, model task_id=43, flip_num=0, model_id=6, first_task_id=65535
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.323.578 [callback.cc:91]163618 Notify:notify [HCCL] task fail start.notify taskid:3 streamid:2 retcode:507011
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.324.653 [callback.cc:91]163618 Notify:notify [MindSpore] task fail start.notify taskid:3 streamid:2 retcode:507011
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.324.699 [stream.cc:1041]163618 GetError:Stream Synchronize failed, stream_id=17, retCode=0x91, [the model stream execute failed].
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.324.711 [stream.cc:1044]163618 GetError:report error module_type=7, module_name=EE9999
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.324.720 [stream.cc:1044]163618 GetError:Task execute failed, base info: device_id=0, stream_id=2, task_id=3, flip_num=0, task_type=3.
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.324.788 [logger.cc:314]163618 StreamSynchronize:Stream synchronize failed
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.324.830 [api_c.cc:721]163618 rtStreamSynchronize:ErrCode=507011, desc=[the model stream execute failed], InnerCode=0x7150050
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.324.841 [error_message_manage.cc:49]163618 FuncErrorReason:report error module_type=3, module_name=EE8888
[ERROR] RUNTIME(160186,python):2023-01-10-09:25:45.324.855 [error_message_manage.cc:49]163618 FuncErrorReason:rtStreamSynchronize execute failed, reason=[the model stream execute failed]
[CRITICAL] GE(160186,fffdf7fff0f0,python):2023-01-10-09:25:45.337.038 [mindspore/ccsrc/plugin/device/ascend/hal/device/ge_runtime/runtime_model.cc:241] Run] Call rt api rtStreamSynchronize failed, ret: 507011
[ERROR] DEVICE(160186,fffdf7fff0f0,python):2023-01-10-09:25:45.337.259 [mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc:761] DumpTaskExceptionInfo] Task fail infos task_id: 3, stream_id: 2, tid: 160186, device_id: 0, retcode: 507011 ( model execute failed)
[CRITICAL] DEVICE(160186,fffdf7fff0f0,python):2023-01-10-09:25:45.337.775 [mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_graph_executor.cc:214] RunGraph] Run task error!Ascend Error Message:EE9999: Inner Error, Please contact support engineer!
EE9999  Task execute failed, base info: device_id=0, stream_id=2, task_id=3, flip_num=0, task_type=3.[FUNC:GetError][FILE:stream.cc][LINE:1044]
        TraceBack (most recent call last):
        rtStreamSynchronize execute failed, reason=[the model stream execute failed][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:49]

[WARNING] MD(160186,ffff9339d010,python):2023-01-10-09:25:45.342.881 [mindspore/ccsrc/minddata/dataset/engine/datasetops/data_queue_op.cc:93] ~DataQueueOp] preprocess_batch: 10; batch_queue: 0, 0, 0, 0, 0, 0, 0, 0, 2, 2; push_start_time: 2023-01-10-09:15:46.739.740, 2023-01-10-09:15:46.748.507, 2023-01-10-09:15:46.772.387, 2023-01-10-09:15:46.772.965, 2023-01-10-09:15:46.775.373, 2023-01-10-09:15:46.779.644, 2023-01-10-09:15:46.787.036, 2023-01-10-09:15:46.791.382, 2023-01-10-09:15:46.795.615, 2023-01-10-09:15:46.797.560; push_end_time: 2023-01-10-09:15:46.742.372, 2023-01-10-09:15:46.749.065, 2023-01-10-09:15:46.772.929, 2023-01-10-09:15:46.773.347, 2023-01-10-09:15:46.775.811, 2023-01-10-09:15:46.780.110, 2023-01-10-09:15:46.787.544, 2023-01-10-09:15:46.791.876, 2023-01-10-09:15:46.796.089, 2023-01-10-09:15:46.797.952.
[TRACE] HCCL(160186,python):2023-01-10-09:25:45.806.979 [status:stop] [hcom.cc:336][hccl-160186-0-1673313336-hccl_world_group][0]hcom destroy complete,take time [86560]us, rankNum[8], rank[0]
Traceback (most recent call last):
  File "train.py", line 36, in <module>
    dataset_base.op_network_with_tdt(ds, epoch_num=beyond_epoch_num)
  File "/home/jenkins/solution_test/common/ms_aw/function/data_processing/dataset_func.py", line 375, in op_network_with_tdt
    assert_iter_num=assert_iter_num)
  File "/home/jenkins/solution_test/common/ms_aw/function/data_processing/dataset_func.py", line 301, in op_network_with_create_dict_iterator
    _ = network()[0]
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 618, in __call__
    out = self.compile_and_run(*args)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 1007, in compile_and_run
    return _cell_graph_executor(self, *new_inputs, phase=self.phase)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 1192, in __call__
    return self.run(obj, *args, phase=phase)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 1229, in run
    return self._exec_pip(obj, *args, phase=phase_real)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 98, in wrapper
    results = fn(*arg, **kwargs)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 1211, in _exec_pip
    return self._graph_executor(args, phase)
RuntimeError: Run task error!

----------------------------------------------------
- Ascend Error Message:
----------------------------------------------------
EE9999: Inner Error, Please contact support engineer!
EE9999  Task execute failed, base info: device_id=0, stream_id=2, task_id=3, flip_num=0, task_type=3.[FUNC:GetError][FILE:stream.cc][LINE:1044]
        TraceBack (most recent call last):
        rtStreamSynchronize execute failed, reason=[the model stream execute failed][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:49]

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_graph_executor.cc:214 RunGraph
```

Generally, this kind of error may be caused by the GetNext timeout for fetching data or framework problems such as control flow and flow assignment. Users can first check whether the problem is caused by the GetNext timeout for fetching data. Some of the following cases may cause GetNext timeout for fetching data:

1. The amount of consumption data in the user script is greater than the amount of production data.

2. When executing the multi-device distributed program, the process of one card may hang, which may cause the timeout of other devices.

If the problem is not caused by GetNext timeout, users can submit issues to [MindSpore Community](https://gitee.com/mindspore) for help.

### EE1001: Device ID Setting Error

Users can specify which card their application runs on by using the environment variable DEVICE_ID or by setting the device_id in the context. If the device id is not set correctly, it may report `EE1001` error, such as the following error scenario. There are only 8 cards in the server, the available device id range is [0, 8), and the user incorrectly set the device_id=8.

```c++
Traceback (most recent call last):
  File "train.py", line 379, in <module>
    train_net()
  File "/home/jenkins/ResNet/scripts/train/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
    run_func(*args, **kwargs)
  File "train.py", line 312, in train_net
    net = resnet(class_num=config.class_num)
  File "/home/jenkins/ResNet/scripts/train/src/resnet.py", line 561, in resnet50
    class_num)
  File "/home/jenkins/ResNet/scripts/train/src/resnet.py", line 381, in __init__
    super(ResNet, self).__init__()
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 118, in __init__
    init_pipeline()
RuntimeError: Device 8 call rtSetDevice failed, ret[107001]. The details refer to 'Ascend Error Message'.

----------------------------------------------------
- Ascend Error Message:
----------------------------------------------------
EE1001: The argument is invalid.Reason: Set device failed, invalid device, set device=8, valid device range is [0, 8)
        TraceBack (most recent call last):
        rtSetDevice execute failed, reason=[device id error][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:49]

(Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)
```

When you encounter such problem, just check the device id setting according to the error log.

## HCCL & HCCP FAQ

HCCL is the Huawei Collective Communication Library, which provides high-performance collective communication functions between servers in deep learning training scenarios, and its communication process is divided into the following three stages:

1. Communication initialization: Obtain the necessary configuration of the aggregate communication parameters and initialize the network devices. The initialization phase does not involve any previous interaction between different devices.

2. Establish communication connections: Establishing a socket connection and exchanging communication parameters and memory information between the two communication ends. During the establishment of the communication connection phase, HCCL builds a link with other cards in combination with the network topology and exchanges information about the parameters used for communication based on the cluster information provided by the user. If no timely response is received from other cards within the build timeout threshold (MindSpore default setting of 600s, configurable via the environment variable HCCL_CONNECT_TIMEOUT), a build timeout error is reported and training is exited.

3. Perform communication operations: Synchronize device execution state and pass memory data via Notify. In the communication operation execution stage, HCCL will schedule NOTIFY/SDMA and other tasks according to the communication algorithm and send them to the task scheduler of the Ascend device through runtime, and the device will schedule and execute the task according to the scheduling information. The Notify class task is used for inter-card synchronization. Notify wait blocks the task stream until the corresponding Notify record arrives, to ensure that each other's memory is in a ready state when subsequent communication operations are performed.

The error codes for HCCL & HCCP start with `EI` and `EJ`. Throughout the communication process, single card problems and communication link problems in the cluster may lead to a large number of timeout errors, so when locating cluster communication problems, we need to collect the log information of the whole cluster and lock the location where the problems occur.

### EI0006: Socket Build Timeout

When the socker build timeout occurs, it will report an `EI0006` error, and the MindSpore log will show a `Distribute Task Failed` error. This means the cluster has a socket build timeout error, as shown in the following log:

```c++
[ERROR] ASCENDCL(83434,python):2022-11-30-23:31:08.729.325 [tensor_data_transfer.cpp:899]89062 acltdtSendTensor: [Push][Data]failed to send, tdt result = -1, device is 1, name is 62576f78-70c2-11ed-b633-000132214e48
[WARNING] DEVICE(83434,fffcf1ffb1e0,python):2022-11-30-23:31:08.986.720 [mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_data_queue.cc:257] Push] Device queue thread had been interrupted by TdtHandle::DestroyHandle, you can ignore the above error: 'failed to send...'. In this scenario, the training ends first without using all epoch(s) data, and the data preprocessing is blocked by the data transmission channel on the device side. So we force the data transmission channel to stop.
[WARNING] MD(83434,ffff852cf5d0,python):2022-11-30-23:31:08.999.090 [mindspore/ccsrc/minddata/dataset/engine/datasetops/data_queue_op.cc:93] ~DataQueueOp] preprocess_batch: 49; batch_queue: 0, 0, 0, 0, 0, 0, 0, 0, 0, 64; push_start_time: 2022-11-30-23:19:41.234.869, 2022-11-30-23:19:41.273.919, 2022-11-30-23:19:41.333.753, 2022-11-30-23:19:41.415.529, 2022-11-30-23:19:41.479.177, 2022-11-30-23:19:41.557.576, 2022-11-30-23:19:41.605.967, 2022-11-30-23:19:41.682.957, 2022-11-30-23:19:41.719.645, 2022-11-30-23:19:41.785.832; push_end_time: 2022-11-30-23:19:41.245.668, 2022-11-30-23:19:41.284.989, 2022-11-30-23:19:41.344.248, 2022-11-30-23:19:41.430.124, 2022-11-30-23:19:41.491.263, 2022-11-30-23:19:41.569.235, 2022-11-30-23:19:41.624.471, 2022-11-30-23:19:41.700.708, 2022-11-30-23:19:41.735.413, 2022-11-30-23:31:08.986.853.
[TRACE] HCCL(83434,python):2022-11-30-23:31:10.455.138 [status:stop] [hcom.cc:264][hccl-83434-0-1669821563-hccl_world_group][1]hcom destroy complete,take time [323391]us, rankNum[8], rank[1]
Traceback (most recent call last):
  File "train.py", line 377, in <module>
    train_net()
  File "/home/jenkins/solution_test/remaining/test_scripts/process/train_parallel1/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
    run_func(*args, **kwargs)
  File "train.py", line 370, in train_net
    sink_size=100, dataset_sink_mode=dataset_sink_mode)
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
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 945, in compile_and_run
    self.compile(*inputs)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 919, in compile
    jit_config_dict=self._jit_config_dict)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 1347, in compile
    result = self._graph_executor.compile(obj, args_list, phase, self._use_vm_mode())
RuntimeError: Preprocess failed before run graph 1. The details refer to 'Ascend Error Message'.

----------------------------------------------------
- Ascend Error Message:
----------------------------------------------------
EI0006: Getting socket times out. Reason: 1. The remote does not initiate a connect request. some NPUs in the cluster are abnormal.    2. The remote does not initiate a connect request because the collective communication operator is started too late or is not started by some NPU in the cluster.    3. The communication link is disconnected. (For example, the IP addresses are not on the same network segment or the TLS configurations are inconsistent.)
        Solution: 1. Check the rank service processes with other errors or no errors in the cluster.2. If this error is reported for all NPUs, check whether the time difference between the earliest and latest errors is greater than the connect timeout interval (120s by default). If so, adjust the timeout interval by using the HCCL_CONNECT_TIMEOUT environment variable.3. Check the connectivity of the communication link between nodes. (For details, see the TLS command and HCCN connectivity check examples.)

(Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)

----------------------------------------------------
- Framework Error Message: (For framework developers)
----------------------------------------------------
Distribute Task Failed,
error msg: davinci_model : load task fail, return ret: 1343225860
```

As stated in the log, the common reasons for socket build timeouts are:

1. Some cards were not executed to the correct building stage, and the error has occurred before the stage.

2. Some cards are blocked by certain tasks that take longer than 600 seconds (configurable via HCCL_CONNECT_TIMEOUT) before the corresponding phase is executed.

3. The communication link between nodes is not working or is unstable.

After collecting INFO logs (including CANN logs) for all cards in the cluster, the following steps can be followed to troubleshoot:

1. Check the error logs of all cards. If a card does not report a socket build timeout error, you can use the log time check to determine whether there is a business process error exit, jamming or core downtime that causes the cluster socket build timeout, and then turn to a single card to locate the problem.

2. If all cards report socket build timeout errors, check whether the difference between the earliest and latest time in the error log of each card exceeds the timeout threshold. If the threshold is exceeded, please locate the latest rank execution blocking cause or adjust the timeout threshold (default is 600 seconds in MindSpore, set by environment variable HCCL_CONNECT_TIMEOUT)

3. Check whether a Device network port communication link in the cluster works. The common reasons are:

    a. IP is not in the same network segment or there is a problem with the subnet mask configuration.

    b. IP conflict. There are two ranks with the same IP in the cluster.

    c. The TLS (security enhancement) settings are inconsistent across ranks.

### EI0002: notify wait timeout

Commonly used in the execution phase. The task of the HCCL operator is executed on each Device of the specified cluster, and the state is synchronized via notify. If an exception occurs on any card or communication link before/during the execution, the cluster synchronization will fail and the remaining cards will have a notify wait timeout and report an `EI0002` error as follows:

```c++
[ERROR] ASCENDCL(162844,python):2022-12-01-00:26:58.086.834 [tensor_data_transfer.cpp:899]168498 acltdtSendTensor: [Push][Data]failed to send, tdt result = -1, device is 1, name is 1393be34-70ca-11ed-9be5-000132214e48
[WARNING] DEVICE(162844,fffce77fe1e0,python):2022-12-01-00:26:58.388.563 [mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_data_queue.cc:257] Push] Device queue thread had been interrupted by TdtHandle::DestroyHandle, you can ignore the above error: 'failed to send...'. In this scenario, the training ends first without using all epoch(s) data, and the data preprocessing is blocked by the data transmission channel on the device side. So we force the data transmission channel to stop.
[CRITICAL] DEVICE(162844,fffd6cff91e0,python):2022-12-01-00:26:58.399.787 [mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_graph_executor.cc:240] RunGraph] Run task for graph:kernel_graph_1 error! The details refer to 'Ascend Error Message'.Ascend Error Message:EI0002: The wait execution of the Notify register times out. Reason: The Notify register has not received the Notify record from remote rank [0].base information: [streamID:[14], taskID[4], taskType[Notify Wait], tag[HcomAllReduce_6629421139219749105_0].] task information: [notify id:[0x0000000100000058], stage:[ffffffff], remote rank:[0].
there are(is) 1 abnormal device(s):
        serverId[10.*.*.*], deviceId[0], Heartbeat Lost Occurred, Possible Reason: 1. Process has exited, 2. Network Disconnected
]
        Possible Cause: 1. An exception occurs during the execution on some NPUs in the cluster. As a result, collective communication operation failed.2. The execution speed on some NPU in the cluster is too slow to complete a communication operation within the timeout interval. (default 1800s, You can set the interval by using HCCL_EXEC_TIMEOUT.)3. The number of training samples of each NPU is inconsistent.4. Packet loss or other connectivity problems occur on the communication link.
        Solution: 1. If this error is reported on part of these ranks, check other ranks to see whether other errors have been reported earlier.2. If this error is reported for all ranks, check whether the error reporting time is consistent (the maximum difference must not exceed 1800s). If not, locate the cause or adjust the locate the cause or set the HCCL_EXEC_TIMEOUT environment variable to a larger value.3. Check whether the completion queue element (CQE) of the error exists in the plog(grep -rn 'error cqe'). If so, check the network connection status. (For details, see the TLS command and HCCN connectivity check examples.)4. Ensure that the number of training samples of each NPU is consistent.
        TraceBack (most recent call last):
        Notify wait execute failed, device_id=1, stream_id=14, task_id=4, flip_num=0, notify_id=11[FUNC:GetError][FILE:stream.cc][LINE:921]
        rtStreamSynchronize execute failed, reason=[the model stream execute failed][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:49]

(Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)
[WARNING] MD(162844,fffce77fe1e0,python):2022-12-01-00:26:58.399.793 [mindspore/ccsrc/minddata/dataset/engine/datasetops/data_queue_op.cc:257] SendDataToAscend] Thread has already been terminated.
malloc_consolidate(): invalid chunk size
```

As stated in the log, common reasons for notify wait timeouts are:

1. Some cards were not executed to the notify synchronization stage, and the error has occurred before the stage.

2. Some cards are blocked by certain tasks that take longer than 1800 seconds (configurable via HCCL_EXEC_TIMEOUT) before the corresponding phase is executed.

3. Inconsistent sequence of task execution between some cards due to network models, etc.

4. The communication link between nodes is unstable.

After collecting INFO logs (including CANN logs) for all cards in the cluster, the following steps can be followed to troubleshoot:

1. Check the error logs of all cards. If a card does not report a notify wait timeout error, you can check the log time to determine whether there is a business process error exit, stuck or core downtime of the cluster notify wait timeout, and then turn to a single card problem location.

2. If all cards report notify wait timeout errors, check whether the difference between the earliest and latest time in the error log of each card exceeds the timeout threshold. If the threshold is exceeded, please locate the latest rank execution blocking cause (e.g. save checkpoint) or adjust the timeout threshold (default is 1800 seconds, set via the environment variable HCCL_EXEC_TIMEOUT).

3. Check whether there is Device network port communication link instability in the cluster, troubleshoot the Device side logs of all cards, and locate the cause of network packet loss if there is an error cqe print and the time is within the business interval.

### EI0004: Illegal ranktable configuration

The user needs to configure the multi-machine multi-card information needed for distributed training through the ranktable file for HCCL initialization. If the relevant ranktable configuration is illegal, an `EI0004` error will be reported, such as the following error scenario, where two device_id are repeatedly set to 1 in the ranktable, resulting in an illegal configuration.

```c++
[WARNING] HCCL_ADPT(89999,ffffa5a47010,python):2023-01-16-20:43:48.480.465 [mindspore/ccsrc/plugin/device/ascend/hal/hccl_adapter/hccl_adapter.cc:47] GenHcclOptions] The environment variable DEPLOY_MODE is not set. Now set to default value 0

Traceback (most recent call last):
  File "train.py", line 379, in <module>
    train_net()
  File "/home/jenkins/ResNet/scripts/train_parallel0/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
    run_func(*args, **kwargs)
  File "train.py", line 305, in train_net
    set_parameter()
  File "train.py", line 151, in set_parameter
    init()
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/communication/management.py", line 161, in init
    init_hccl()
RuntimeError: Ascend collective communication initialization failed.

----------------------------------------------------
- Ascend Error Message:
----------------------------------------------------
EI0004: The ranktable is invalid,Reason:[The ranktable config devId is inconsistent with the local devId.]. Please check the configured ranktable. [{"server_count":"1","server_list":[{"device":[{"device_id":"1","device_ip":"192.168.100.101","rank_id":"0"},{"device_id":"1","device_ip":"192.168.101.101","rank_id":"1"},{"device_id":"2","device_ip":"192.168.102.101","rank_id":"2"},{"device_id":"3","device_ip":"192.168.103.101","rank_id":"3"},{"device_id":"4","device_ip":"192.168.100.102","rank_id":"4"},{"device_id":"5","device_ip":"192.168.101.102","rank_id":"5"},{"device_id":"6","device_ip":"192.168.102.102","rank_id":"6"},{"device_id":"7","device_ip":"192.168.103.102","rank_id":"7"}],"host_nic_ip":"reserve","server_id":"10.*.*.*"}],"status":"completed","version":"1.0"}]
        Solution: Try again with a valid cluster configuration in the ranktable file. Ensure that the configuration matches the operating environment.

(Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)

----------------------------------------------------
- Framework Error Message: (For framework developers)
----------------------------------------------------
Init hccl graph adapter failed.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.cc:112 Initialize
mindspore/ccsrc/plugin/device/ascend/hal/hccl_adapter/hccl_adapter.cc:402 InitKernelInfoStore
```

When you encounter such an error, just fix the ranktable configuration file according to the error log.

### EI0005: Inconsistent Communication Parameters Between Cards

The `EI0005` error is reported when there are inconsistencies in the communication parameters between cards, such as inconsistencies in the size of the input shape used for AllReduce between cards, as in the following error reporting scenario, where there is a parameter named count that is inconsistent in size when communicating between cards.

```c++
[WARNING] HCCL_ADPT(50288,ffff8fec4010,python):2023-01-16-20:37:22.585.027 [mindspore/ccsrc/plugin/device/ascend/hal/hccl_adapter/hccl_adapter.cc:47] GenHcclOptions] The environment variable DEPLOY_MODE is not set. Now set to default value 0
[WARNING] MD(50288,fffe35f4b0f0,python):2023-01-16-20:38:57.747.318 [mindspore/ccsrc/minddata/dataset/engine/datasetops/source/generator_op.cc:198] operator()] Bad performance attention, it takes more than 25 seconds to generator.__next__ new row, which might cause `GetNext` timeout problem when sink_mode=True. You can increase the parameter num_parallel_workers in GeneratorDataset / optimize the efficiency of obtaining samples in the user-defined generator function.
dataset length:  848
data pre-process time is 0.2904245853424072

Traceback (most recent call last):
  File "e2e_feed_dev.py", line 294, in <module>
    run()
  File "e2e_feed_dev.py", line 277, in run
    label_indices, label_values)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 626, in __call__
    out = self.compile_and_run(*args)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 945, in compile_and_run
    self.compile(*inputs)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/nn/cell.py", line 919, in compile
    jit_config_dict=self._jit_config_dict)
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/common/api.py", line 1337, in compile
    result = self._graph_executor.compile(obj, args_list, phase, self._use_vm_mode())
RuntimeError: Preprocess failed before run graph 0. The details refer to 'Ascend Error Message'.

----------------------------------------------------
- Ascend Error Message:
----------------------------------------------------
EI0005: The arguments for collective communication are inconsistent between ranks: tag [HcomAllReduce_6629421139219749105_0], parameter [count], local [9556480], remote [9555712]
        Solution: Check whether the training script and ranktable of each NPU are consistent.

(Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)

----------------------------------------------------
- Framework Error Message: (For framework developers)
----------------------------------------------------
Distribute Task Failed,
error msg: davinci_model : load task fail, return ret: 1343225860

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_kernel_executor.cc:206 PreprocessBeforeRunGraph
mindspore/ccsrc/plugin/device/ascend/hal/device/ascend_kernel_runtime.cc:578 LoadTask
mindspore/ccsrc/plugin/device/ascend/hal/device/ge_runtime/task/hccl_task.cc:104 Distribute
```

When such errors are occurred, the communication parameters in error can be identified through logs and IR diagrams and corrected in the network script.

### EJ0001: HCCP Initialization Failure

The HCCP process is responsible for implementing the communication function, and HCCL can call the HCCP interface for communication. HCCP initialization failure will report `EJ0001` error, such as the following scenario. When the previous eight-card training task has not yet finished in the same server to start a new eight-card training task, initialization failure will occur. You need to wait for the previous eight-card training task to finish before starting a new eight-card training task.

```c++
[ERROR] HCCL(17381,python):2022-12-01-03:02:29.001.054 [network_manager.cc:64][hccl-17381-0-1669834948-hccl_world_group][0]call trace: ret -> 7
[ERROR] HCCL(17381,python):2022-12-01-03:02:29.001.066 [hccl_impl_base.cc:239][hccl-17381-0-1669834948-hccl_world_group][0]call trace: ret -> 7
[ERROR] HCCL(17381,python):2022-12-01-03:02:29.001.076 [hccl_impl.cc:731][hccl-17381-0-1669834948-hccl_world_group][0]call trace: ret -> 7
[ERROR] HCCL(17381,python):2022-12-01-03:02:29.001.104 [hccl_comm.cc:100][hccl-17381-0-1669834948-hccl_world_group][0][HcclComm][Init]errNo[0x0000000005000007] hccl initialize failed
[ERROR] HCCL(17381,python):2022-12-01-03:02:29.001.116 [hcom.cc:80][hccl-17381-0-1669834948-hccl_world_group][0][Init][Result]errNo[0x0000000005010007] hcclComm init error
[ERROR] HCCL(17381,python):2022-12-01-03:02:29.001.129 [hcom.cc:94][hccl-17381-0-1669834948-hccl_world_group][0][Init][Result]hcom init failed, rankNum[8], rank[0], server[10.*.*.*], device[0], return[83951623]
[TRACE] HCCL(17381,python):2022-12-01-03:02:29.001.628 [status:stop] [hcom.cc:264][hccl-17381-0-1669834948-hccl_world_group][0]hcom destroy complete,take time [486]us, rankNum[8], rank[0]
[ERROR] HCCL(17381,python):2022-12-01-03:02:29.001.645 [hcom.cc:190][hccl-17381-0-1669834948-hccl_world_group][0][HcomInitByFile]errNo[0x0000000005000007] rankTablePath[/ms_test/workspace/config/hccl_8p.json] identify[0] hcom init failed.
[ERROR] HCCL(17381,python):2022-12-01-03:02:29.001.662 [hcom_plugin.cc:202][hccl-17381-0-1669834948-hccl_world_group][0][Init][HcomPlugin]errNo[0x0000000005010007] Initialize: HcomInitByFile failed.
[ERROR] HCCL(17381,python):2022-12-01-03:02:29.001.675 [hcom_plugin.cc:61][hccl-17381-0-1669834948-hccl_world_group][0][Initialize][Plugin]Initialize Hcom failed
[CRITICAL] HCCL_ADPT(17381,ffff890145d0,python):2022-12-01-03:02:29.001.753 [mindspore/ccsrc/plugin/device/ascend/hal/hccl_adapter/hccl_adapter.cc:402] InitKernelInfoStore] Init hccl graph adapter failed.
[CRITICAL] DEVICE(17381,ffff890145d0,python):2022-12-01-03:02:29.002.576 [mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.cc:112] Initialize] Ascend collective communication initialization failed.Ascend Error Message:EJ0001: Failed to initialize the HCCP process. Reason: Maybe the last training process is running.
        Solution: Wait for 10s after killing the last training process and try again.
        TraceBack (most recent call last):
        tsd client wait response fail, device response code[1]. unknown device error.[FUNC:WaitRsp][FILE:process_mode_manager.cpp][LINE:233]

(Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)Framework Error Message:Init hccl graph adapter failed.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/device/ascend/hal/hccl_adapter/hccl_adapter.cc:402 InitKernelInfoStore

[CRITICAL] DEVICE(17381,ffff890145d0,python):2022-12-01-03:02:29.011.280 [mindspore/ccsrc/runtime/device/kernel_runtime.cc:124] LockRuntime] The pointer[stream] is null.
[ERROR] DEVICE(17381,ffff890145d0,python):2022-12-01-03:02:29.011.608 [mindspore/ccsrc/runtime/device/kernel_runtime_manager.cc:138] WaitTaskFinishOnDevice] SyncStream failed, exception:The pointer[stream] is null.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/runtime/device/kernel_runtime.cc:124 LockRuntime


[ERROR] RUNTIME(17381,python):2022-12-01-03:02:29.091.105 [engine.cc:1044]17901 ReportStatusFailProc:Device status failure, ret=118554641,start exception CallBack.
[ERROR] DRV(17381,python):2022-12-01-03:02:29.391.021 [ascend][curpid: 17381, 17381][drv][tsdrv][share_log_read 552]hdc connect down, devid(0) fid(0) tsid(0) hdc connect down, devid(0) fid(0) tsid(0)
Traceback (most recent call last):
  File "train.py", line 377, in <module>
    train_net()
  File "/home/jenkins/workspace/TDT_deployment/solution_test/remaining/test_scripts/mindspore/reliability/fmea/business/process/multitask/test_ms_fmea_multi_task_two_8p_0001_2_GRAPH_MODE/scripts/train_parallel0/src/model_utils/moxing_adapter.py", line 104, in wrapped_func
    run_func(*args, **kwargs)
  File "train.py", line 304, in train_net
    set_parameter()
  File "train.py", line 151, in set_parameter
    init()
  File "/home/miniconda3/envs/ci/lib/python3.7/site-packages/mindspore/communication/management.py", line 152, in init
    init_hccl()
RuntimeError: Ascend collective communication initialization failed.

----------------------------------------------------
- Ascend Error Message:
----------------------------------------------------
EJ0001: Failed to initialize the HCCP process. Reason: Maybe the last training process is running.
        Solution: Wait for 10s after killing the last training process and try again.
        TraceBack (most recent call last):
        tsd client wait response fail, device response code[1]. unknown device error.[FUNC:WaitRsp][FILE:process_mode_manager.cpp][LINE:233]

(Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)

----------------------------------------------------
- Framework Error Message: (For framework developers)
----------------------------------------------------
Init hccl graph adapter failed.

----------------------------------------------------
- C++ Call Stack: (For framework developers)
----------------------------------------------------
mindspore/ccsrc/plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.cc:112 Initialize
mindspore/ccsrc/plugin/device/ascend/hal/hccl_adapter/hccl_adapter.cc:402 InitKernelInfoStore
```

When you encounter such an error, just solve it according to the initialization error cause and solution of the log.

## profiling FAQ

The profiling issue error code starts with `EK`.

### EK0001: Illegal Parameter Problem

The `EK0001` error is reported when CANN's profiling module interface encounters an illegal parameter. The most common illegal parameter error for users is the incorrect path setting for profiling, such as the following error:

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

From the error message of CANN in ERROR log, we can know that the path character of profiling can only contain `A-Za-z0-9-_`, and the path of profiling in the above error contains `Chinese` characters, which leads to illegal path error, thus causing profiling initialization failure. If you encounter such problem, you can correct the profiling path or other parameters according to the error message.
