# Release Notes

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0rc1/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/mindspore/blob/v2.7.0-rc1/RELEASE.md)

## MindSpore 2.7.0-rc1 Release Notes

### Major Features and Improvements

#### Ascend

- [STABLE] Added GroupedMatmul optimization in O1 scenario, which supports fusion of element-wise operators to significantly reduce data handling overhead and improve computational efficiency. Users can turn it on by setting the environment variable `MS_DEV_GRAPH_KERNEL_FLAGS` to `"--enable_cluster_ops=GroupedMatmul"`.
- [STABLE] Improved ease of use of memory tracker: Users can import tracker data for model memory analysis via [mindspore.runtime.memory_replay(file_path)](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/runtime/mindspore.runtime.memory_replay.html); set the tracker data storage path by setting `export MS_ALLOC_CONF=memory_tracker_path:file_path`；reduce the size of saved data by setting `export MS_ALLOC_CONF=simple_tracker:true` to save only the last user of each memory block.
- [STABLE] Optimized the custom operator function of ops.Custom primitive for aclnn types in graph mode, with full support for inputs of non-Tensor type and automatic loading of RegInfo information, which significantly improves the ease of use and flexibility of aclnn custom operator.

#### PyNative

- [STABLE] [mindspore.nn.Cell.register_forward_hook](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_hook) and [mindspore.nn.Cell.register_forward_pre_hook](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_pre_hook) added the `with_kwargs` argument (default: False) to support passing keyword arguments from the `construct` call to `hook_fn`.
- [STABLE] [mindspore.Tensor.register_hook](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mindspore/Tensor/mindspore.Tensor.register_hook.html#mindspore.Tensor.register_hook) now supports registering hooks on output tensors of operators with multiple outputs.
- [STABLE] Enhanced Cell custom bprop. Support verification and automatic conversion of data type and shape for return value and corresponding input.
- [STABLE] Added [Storage](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mindspore/Tensor/mindspore.Tensor.storage.html) API for Tensor: Supports memory operations through Storage and related interfaces to achieve memory optimization.
- [STABLE] Added C++ [ms::pynative::PyboostRunner](https://www.mindspore.cn/tutorials/en/r2.7.0rc1/custom_program/operation/cpp_api_for_custom_ops.html#class-pyboostrunner) interface to facilitate customization of the operator to support PyNative's multi-stage pipelining runtime.

#### parallel

- [STABLE] Configurable HCCL buffer size for communucation group: [communication.create_group](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/communication/mindspore.communication.create_group.html#mindspore.communication.create_group) supports additional configuration `options` in passing communication domain, sets the communication domain HCCL communication domain cache size by `hccl_config` key-value pair to avoid OOM under multiple communication domains.
- [STABLE] Supported CPU communication interfaces [mint.distributed.all_reduce](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mint/mindspore.mint.distributed.all_reduce.html)/[mint.distributed.barrier](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mint/mindspore.mint.distributed.barrier.html)/[mint.distributed.send](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mint/mindspore.mint.distributed.send.html)/[mint.distributed.recv](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mint/mindspore.mint.distributed.recv.html), through which users can use the corresponding collective communication operators functions.
- [STABLE] In static graph mode, forward and reverse operators, [AllGatherV](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/ops/mindspore.ops.AllGatherV.html) and [ReduceScatterV](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/ops/mindspore.ops.ReduceScatterV.html) are supported.
- [BETA] Support for data caching via member methods of [mint.distributed.TCPStore](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mint/mindspore.mint.distributed.TCPStore.html) function.

#### Training

- [STABLE] Recomputation communication overlap: Supports mutual overlap of the communication between two cells for full recomputation, improving the performance of the recomputation scenario model.
- [STABLE] Quick recovery from accuracy failure without reboot: Supports resuming training of loading a checkpoint file without restrarting the process when training result excepition occurs.
- [STABLE] Silent data corruption detection: Supporting validation of `MatMul` results during forward and backward computations. User can enable it by setting the environment variable `MS_SDC_DETECT_ENABLE` to `1`. Using interfaces in the `mindspore.utils.sdc_detect` module to start/stop detection and get the detection result.

#### Inference

- [STABLE] The vLLM MindSpore plugin has now been adapted to vLLM v0.8.3 and supports foundational features of the vLLM V1 new architecture, including inference capabilities such as Chunked Prefill and Automatic Prefix Caching. For service-oriented deployment, vLLM MindSpore adds support for hybrid DP/TP/EP parallel inference capabilities on DeepSeek-V3/R1, effectively improving both full and incremental inference efficiency while reducing device memory overhead.

#### Tool

MindInsight will no longer update or release new versions after version 2.3，and the related documents have been removed. The origin system optimization data visualization has been integrated into MindStudio Insight, and scalar visualization, parameter distribution visualization, and computational graphs visualization have been integrated into the MindStudio Insight plugins. For details, see the [MindStudio Insight User Guide](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html).

- [STABLE] MindSpore Profiler supports msMonitor, enabling users to collect performance data through online monitoring tools.
- [STABLE] MindSpore Profiler adds the record_shapes parameter, supporting users to collect shapes of operators issued by the framework side.
- [STABLE] MindSpore Profiler adds sys resource parameters, supporting the ability to collect sys resource class information.
- [STABLE] MindSpore Profiler adds host_sysparameter, supporting the ability to collect host information such as system call class, storage class, CPU information, etc.
- [STABLE] MindSpore Profiler mstx module provides domain functionality, supporting users to finely control mstx data.

### API Change

- [STABLE] Some of the functional, nn and Tensor interfaces in the DEMO state in the [mindspore.mint](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mindspore.mint.html) API are turned to STABLE. The mint interfaces are still mostly experimental, with better performance than ops interfaces in the graph compilation mode O0/O1 and PyNative mode. Currently O2 compilation mode (graph sink) and CPU, GPU backend are not supported, which will be gradually improved.

  | mindspore.mint              |
  | :-------------------------- |
  | mindspore.mint.randomperm   |
  | mindspore.mint.randn        |
  | mindspore.mint.randint      |
  | mindspore.mint.triu         |
  | mindspore.mint.empty_like   |
  | mindspore.mint.empty        |
  | mindspore.mint.floor_divide |

  | mindspore.mint.nn              |
  | :----------------------------- |
  | mindspore.mint.nn.BatchNorm1d  |
  | mindspore.mint.nn.BatchNorm2d  |
  | mindspore.mint.nn.BatchNorm3d  |
  | mindspore.mint.nn.PixelShuffle |
  | mindspore.mint.nn.Threshold    |

  | mindspore.mint.nn.functional               |
  | :----------------------------------------- |
  | mindspore.mint.nn.functional.threshold     |
  | mindspore.mint.nn.functional.threshold_    |
  | mindspore.mint.nn.functional.pixel_shuffle |

  | mindspore.Tensor              |
  | ----------------------------- |
  | mindspore.Tensor.new_full     |
  | mindspore.Tensor.new_empty    |
  | mindspore.Tensor.floor_divide |
  | mindspore.Tensor.exponential_ |

- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mindspore.ops.html) API provides an new interface [mindspore.ops.swiglu](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/ops/mindspore.ops.swiglu.html). Currently, only Ascend backend is supported.
- [STABLE] [mindspore.ops.svd](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/ops/mindspore.ops.svd.html#mindspore.ops.svd) of [mindspore.ops](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mindspore.ops.html) API now extra supports Ascend backend.
- [STABLE] [mindspore.mint.nn.functional.silu](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mint/mindspore.mint.nn.functional.silu.html#mindspore.mint.nn.functional.silu) and [mindspore.mint.nn.SiLU](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mint/mindspore.mint.nn.SiLU.html) now support input argument `inplace`.
- [STABLE] [communication.create_group](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/communication/mindspore.communication.create_group.html#mindspore.communication.create_group) adds support for additional configuration `options` for communication domains. HCCL backend supports setting `hccl_config` in `options` to set the HCCL communication domain cache size for communication domains.
- [STABLE] [mindspore.runtime](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/mindspore.runtime.html) API adds implementation of [mindspore.runtime.empty_cache](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/runtime/mindspore.runtime.empty_cache.html#mindspore.runtime.empty_cache).
- [STABLE] [mindspore.runtime.set_memory](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/runtime/mindspore.runtime.set_memory.html#mindspore.runtime.set_memory) now supports input argument `huge_page_reserve_size`.
- [STABLE] [mindspore.runtime.set_cpu_affinity](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/runtime/mindspore.runtime.set_cpu_affinity.html) now supports input argument `module_to_cpu_dict`.
- [STABLE] [minspore.nn.cell](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/nn/mindspore.nn.Cell.html) added the function to view/save model's state_dict. New interfaces are as follows:

  | mindspore.nn.Cell                       |
  | --------------------------------------- |
  | cell.register_state_dict_post_hook      |
  | cell.register_state_dict_pre_hook       |
  | cell.state_dict                         |
  | cell.register_load_state_dict_pre_hook  |
  | cell.register_load_state_dict_post_hook |
  | cell.load_state_dict                    |

- [STABLE] [minspore.nn.cell](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/nn/mindspore.nn.Cell.html) added the function to view/register model's buffer. New interfaces are as follows:

  | mindspore.nn.Cell    |
  | -------------------- |
  | cell.register_buffer |
  | cell.get_buffer      |
  | cell.get_sub_cell    |
  | cell.named_buffer    |
  | cell.buffers         |

#### Backwards Incompatible Change

- [runtime.set_cpu_affinity](https://www.mindspore.cn/docs/en/r2.7.0rc1/api_python/runtime/mindspore.runtime.set_cpu_affinity.html)

  The type of `affinity_cpu_list` changed from dictionary to list to customize the configuration of affinity CPU range segments for a single process only. Added a new parameter `module_to_cpu_dict` to support customized configuration of CPU affinity policies for hot module threads.

  | 2.6     | 2.7       |
  | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
  | >>> from mindspore.runtime import set_cpu_affinity<br>>>> set_cpu_affinity(True, {"device0": ["10-19", "23-40"]}) | >>> from mindspore.runtime import set_cpu_affinity<br>>>> set_cpu_affinity(True, ["10-19", "23-40"],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {"main": [0,1,2,3],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "runtime": [4,5,6],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "pynative": [7,8,9]}) |

### Contributors

baochong,Bellatan,BJ-WANG,caifubi,caiji_zhang,Carey,chaijinwei,changzherui,chengbin,chujinjin,DavidFFFan,DeshiChen,dingjinshan,Dring,ehaleva,Erpim,fary86,fengtingyan,fengyixing,fuchao,gaoyong10,gengdongjie,guangpengz,GuoZhibin,gupengcheng0401,haozhang,hedongdong,hhz886,huandong1,huangbingjian,huangziling,huda,HuilanLi,hujiahui8,jiangchao_j,jianghui58,jiangshanfeng,jiaorui,jiaxueyu,jizewei,jjfeing,jshawjc,kairui_kou,kingxian,kisnwang,lanzhineng,leida,LiangZhibo,lichen,limingqi107,LiNuohang,linux,liubuyu,liuchengji,liuluobin,liuyanwei,lkp,looop5,lujiale,luochao60,luoyang,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,mengyuanli,NaCN,One_East,panshaowu,panzhihui,pengqi,Qiao_Fu,qiuleilei,qiuyufeng,rainyhorse,SaiYao,shaoshengqi,shen_haochen,shenhaojing,shenwei41,shuqian0,tanghuikang,tangmengcheng,tan-wei-cheng,tianxiaodong,uuhuu,wang_ziqi,WangChengzhao,wangshaocong,wangyibo,wujueying,XianglongZeng,xiaopeng,xiaotianci,xiaoyao,XinDu,xuzhen,yangguodong,yanghaoran,yangyingchun,Yanzhi_YI,yide12,yihangchen,YijieChen,yuanqi,yuchaojie,yuezenglin,YuJianfeng,YukioZzz,yuliangbin,yyuse,zhangbuxue,zhangdanyang,zhanghanLeo,zhangyinxia,ZhangZGC,zhanzhan,zhaochenjie,zhengzuohe,zhouyaqiang0,zhunaipan,zichun_ye,ZPaC,zyli2020,程超,范吉斌,胡犇,胡彬,宦晓玲,黄勇,李栋,李良灿,李林杰,李寅杰3,刘飞扬,刘力力,刘勇琪,刘子涵,梅飞要,宋佳琪,王泓皓,王禹程,王振邦,熊攀,徐安越,杨卉,杨明海,俞涵,虞良斌,云骑士,张栩浩,周一航
