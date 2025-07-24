# Release Notes

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.7.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/mindspore/blob/v2.7.0-rc1/RELEASE_CN.md)

## MindSpore 2.7.0-rc1 Release Notes

### 主要特性及增强

#### Ascend

- [STABLE] 新增 O1 场景下的 GroupedMatmul 优化，支持该算子后融合 element-wise 算子，大幅减少数据搬运开销，提升计算效率。用户可以通过设置环境变量 `MS_DEV_GRAPH_KERNEL_FLAGS`为 `"--enable_cluster_ops=GroupedMatmul"`开启。
- [STABLE] 提升内存 tracker 易用性：用户可以通过 [mindspore.runtime.memory_replay(file_path)](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/runtime/mindspore.runtime.memory_replay.html) 导入 tracker 数据，用于模型显存分析；通过设置 `export MS_ALLOC_CONF=memory_tracker_path:file_path` 设置 tracker 数据保存路径；通过设置 `export MS_ALLOC_CONF=simple_tracker:true` 仅保存每个内存块最后一个使用者，减少保存数据大小。
- [STABLE] 优化了图模式下ops.Custom原语针对aclnn类型的自定义算子功能，全面支持非Tensor类型输入，并自动加载RegInfo信息，显著提升了aclnn自定义算子的易用性和灵活性。

#### PyNative

- [STABLE] [mindspore.nn.Cell.register_forward_hook](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_hook) 和 [mindspore.nn.Cell.register_forward_pre_hook](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.register_forward_pre_hook) 新增 `with_kwargs` 参数（默认值：False），用于支持将 `construct` 调用时的关键字参数传递给 `hook_fn`。
- [STABLE] [mindspore.Tensor.register_hook](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mindspore/Tensor/mindspore.Tensor.register_hook.html#mindspore.Tensor.register_hook) 现已支持对多输出算子的输出 Tensor 注册 hook。
- [STABLE] Cell自定义反向功能增强，支持返回值的类型和对应输入的数据类型和shape形状的校验和自动转换。
- [STABLE] Tensor增加[Storage](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mindspore/Tensor/mindspore.Tensor.storage.html?highlight=storage#mindspore.Tensor.storage)接口：支持通过Storage及其相关接口对内存进行操作，从而实现内存优化。
- [STABLE] 新增 C++ 的 [ms::pynative::PyboostRunner](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0rc1/custom_program/operation/cpp_api_for_custom_ops.html#class-pyboostrunner) 接口，方便自定义算子支持 PyNative 多级流水运行时。

#### 并行

- [STABLE] 针对通信域可配HCCL缓存：[communication.create_group](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/communication/mindspore.communication.create_group.html#mindspore.communication.create_group) 支持传入通信域额外配置 `options`，通过 `hccl_config`键值对设置通信域HCCL通信域缓存大小，避免多通信域下OOM。
- [STABLE] 支持CPU通信接口 [mint.distributed.all_reduce](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.distributed.all_reduce.html)/[mint.distributed.barrier](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.distributed.barrier.html)/[mint.distributed.send](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.distributed.send.html)/[mint.distributed.recv](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.distributed.recv.html)，用户可通过这些接口使用对应的集合通信算子功能。
- [STABLE] 静态图模式场景，支持 [AllGatherV](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/ops/mindspore.ops.AllGatherV.html) 和[ReduceScatterV](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/ops/mindspore.ops.ReduceScatterV.html)正反向算子。
- [BETA] 支持通过[mint.distributed.TCPStore](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.distributed.TCPStore.html)类的成员方法实现数据缓存功能。

#### 训练

- [STABLE] 重计算通信掩盖：支持对于完全重计算的两个Cell的计算通信进行互相掩盖，提升重计算场景模型的性能。
- [STABLE] 精度故障不重启快速恢复：支持当发生精度故障时，在不重启进程的情况下加载CheckPoint继续训练。
- [STABLE] 静默检测：支持对正反向计算中 `MatMul`结果进行校验。用户可以通过设置环境变量 `MS_SDC_DETECT_ENABLE` 为 `1` 使能，并通过 `mindspore.utils.sdc_detect` 模块的接口动态开启/关闭检测并获取结果。

#### 推理

- [STABLE] vLLM MindSpore 插件现已适配 vLLM v0.8.3 版本，并支持 vLLM V1 新架构的基础特性，包含 Chunked Prefill、Automatic Prefix Caching 等推理能力，在服务化部署方面，新增支持运行 DeepSeek-V3/R1  DP/TP/EP 混合并行推理能力，有效提升模型全量及增量推理效率，同时降低显存开销；

#### 工具

自2.3版本起MindInsight工具停止更新发布，相关文档已下架。原有系统调优数据可视化已整合至MindStudio Insight，标量可视化、参数分布图可视化和计算图可视化已整合至MindStudio Insight插件，请参阅[《MindStudio Insight用户指南》](https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/GUI_baseddevelopmenttool/msascendinsightug/Insight_userguide_0002.html)。

- [STABLE] MindSpore Profiler支持接入msMonitor功能，支持用户通过在线监控工具采集性能数据。
- [STABLE] MindSpore Profiler增加record_shapes参数，支持用户采集框架侧下发算子的shape信息。
- [STABLE] MindSpore Profiler增加sys类资源类开关，支持sys资源类信息采集能力。
- [STABLE] MindSpore Profiler增加host_sys开关，支持系统调用类、存储类、cpu信息等host信息采集能力。
- [STABLE] MindSpore Profiler mstx模块提供domain功能，支持用户精细化控制mstx打点数据。

### API 变更

- [STABLE] [mindspore.mint](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mindspore.mint.html) API中部分DEMO状态的functional、nn和Tensor接口转为STABLE。mint接口当前大多仍为实验性接口，在图编译模式为O0/O1和PyNative模式下性能比ops更优。当前暂不支持O2编译模式(图下沉)及CPU、GPU后端，后续会逐步完善。

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

- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mindspore.ops.html) API新增[mindspore.ops.swiglu](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/ops/mindspore.ops.swiglu.html)接口，当前仅支持Ascend后端。
- [STABLE] [mindspore.ops](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mindspore.ops.html) API的[mindspore.ops.svd](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/ops/mindspore.ops.svd.html#mindspore.ops.svd)接口现在额外支持了Ascend后端。
- [STABLE] [mindspore.mint.nn.functional.silu](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.nn.functional.silu.html#mindspore.mint.nn.functional.silu)和 [mindspore.mint.nn.SiLU](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mint/mindspore.mint.nn.SiLU.html) 新增支持了入参 `inplace`。
- [STABLE] [communication.create_group](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/communication/mindspore.communication.create_group.html#mindspore.communication.create_group) 新增支持通信域额外配置 `options`。HCCL后端支持设置 `options`中的 `hccl_config`，针对通信域设置HCCL通信域缓存大小。
- [STABLE] [mindspore.runtime](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/mindspore.runtime.html)增加[mindspore.runtime.empty_cache](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/runtime/mindspore.runtime.empty_cache.html#mindspore.runtime.empty_cache)的实现。
- [STABLE] [mindspore.runtime.set_memory](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/runtime/mindspore.runtime.set_memory.html#mindspore.runtime.set_memory)接口新增入参 `huge_page_reserve_size`。
- [STABLE] [mindspore.runtime.set_cpu_affinity](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/runtime/mindspore.runtime.set_cpu_affinity.html#mindspore.runtime.set_cpu_affinity)接口新增入参 `module_to_cpu_dict`。
- [STABLE] [minspore.nn.Cell](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/nn/mindspore.nn.Cell.html)模块新增查看/保存模型状态的功能。新增接口：

  | mindspore.nn.Cell                       |
  | --------------------------------------- |
  | cell.register_state_dict_post_hook      |
  | cell.register_state_dict_pre_hook       |
  | cell.state_dict                         |
  | cell.register_load_state_dict_pre_hook  |
  | cell.register_load_state_dict_post_hook |
  | cell.load_state_dict                    |

- [STABLE] [minspore.nn.Cell](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/nn/mindspore.nn.Cell.html)模块查看/注册模型缓冲区的功能。新增接口：

  | mindspore.nn.Cell    |
  | -------------------- |
  | cell.register_buffer |
  | cell.get_buffer      |
  | cell.get_sub_cell    |
  | cell.named_buffer    |
  | cell.buffers         |

#### 非兼容性接口变更

- [runtime.set_cpu_affinity](https://www.mindspore.cn/docs/zh-CN/r2.7.0rc1/api_python/runtime/mindspore.runtime.set_cpu_affinity.html)

  参数 `affinity_cpu_list`类型由字典变更为列表，仅针对单一进程自定义配置亲和CPU范围段。新增参数 `module_to_cpu_dict`，支持针对热点模块线程自定义配置CPU亲和策略。

  | 2.6     | 2.7       |
  | ------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
  | >>> from mindspore.runtime import set_cpu_affinity<br>>>> set_cpu_affinity(True, {"device0": ["10-19", "23-40"]}) | >>> from mindspore.runtime import set_cpu_affinity<br>>>> set_cpu_affinity(True, ["10-19", "23-40"],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {"main": [0,1,2,3],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "runtime": [4,5,6],<br> ... &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "pynative": [7,8,9]}) |

### 贡献者

baochong,Bellatan,BJ-WANG,caifubi,caiji_zhang,Carey,chaijinwei,changzherui,chengbin,chujinjin,DavidFFFan,DeshiChen,dingjinshan,Dring,ehaleva,Erpim,fary86,fengtingyan,fengyixing,fuchao,gaoyong10,gengdongjie,guangpengz,GuoZhibin,gupengcheng0401,haozhang,hedongdong,hhz886,huandong1,huangbingjian,huangziling,huda,HuilanLi,hujiahui8,jiangchao_j,jianghui58,jiangshanfeng,jiaorui,jiaxueyu,jizewei,jjfeing,jshawjc,kairui_kou,kingxian,kisnwang,lanzhineng,leida,LiangZhibo,lichen,limingqi107,LiNuohang,linux,liubuyu,liuchengji,liuluobin,liuyanwei,lkp,looop5,lujiale,luochao60,luoyang,maoyuanpeng1,Margaret_wangrui,mengxian,MengXiangyu,mengyuanli,NaCN,One_East,panshaowu,panzhihui,pengqi,Qiao_Fu,qiuleilei,qiuyufeng,rainyhorse,SaiYao,shaoshengqi,shen_haochen,shenhaojing,shenwei41,shuqian0,tanghuikang,tangmengcheng,tan-wei-cheng,tianxiaodong,uuhuu,wang_ziqi,WangChengzhao,wangshaocong,wangyibo,wujueying,XianglongZeng,xiaopeng,xiaotianci,xiaoyao,XinDu,xuzhen,yangguodong,yanghaoran,yangyingchun,Yanzhi_YI,yide12,yihangchen,YijieChen,yuanqi,yuchaojie,yuezenglin,YuJianfeng,YukioZzz,yuliangbin,yyuse,zhangbuxue,zhangdanyang,zhanghanLeo,zhangyinxia,ZhangZGC,zhanzhan,zhaochenjie,zhengzuohe,zhouyaqiang0,zhunaipan,zichun_ye,ZPaC,zyli2020,程超,范吉斌,胡犇,胡彬,宦晓玲,黄勇,李栋,李良灿,李林杰,李寅杰3,刘飞扬,刘力力,刘勇琪,刘子涵,梅飞要,宋佳琪,王泓皓,王禹程,王振邦,熊攀,徐安越,杨卉,杨明海,俞涵,虞良斌,云骑士,张栩浩,周一航
