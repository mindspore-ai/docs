"""utils implementation"""
import os
import mindspore as ms
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.parallel import set_algo_parameters


def set_graph_kernel_context(device_target, net_name):
    """When device_target is GPU and network is resnet101 set `enable_graph_kernel=True`."""
    if device_target == "GPU" and net_name == "resnet101":
        ms.set_context(enable_graph_kernel=True)
        ms.set_context(graph_kernel_flags="--enable_parallel_fusion --enable_expand_ops=Conv2D")


def init_env(cfg):
    """Initialize the runtime environment."""
    ms.set_seed(cfg.seed)

    # If device_target is None, auto select device_target.
    if cfg.device_target != "None":
        if cfg.device_target not in ["Ascend", "GPU", "CPU"]:
            raise ValueError(f"Invalid device_target: {cfg.device_target}, "
                             f"should be in ['None', 'Ascend', 'GPU', 'CPU']")
        ms.set_context(device_target=cfg.device_target)
    cfg.device_target = ms.get_context("device_target")
    if cfg.device_target == "CPU":
        cfg.device_id = 0
        cfg.device_num = 1
    if cfg.context_mode not in ["graph", "pynative"]:
        raise ValueError(f"Invalid context_mode: {cfg.context_mode}, "
                         f"should be in ['graph', 'pynative']")
    context_mode = ms.GRAPH_MODE if cfg.context_mode == "graph" else ms.PYNATIVE_MODE
    if not os.path.isabs(cfg.save_graphs_path):
        cfg.save_graphs_path = os.path.join(cfg.output_path, cfg.save_graphs_path)
    ms.set_context(mode=context_mode, save_graphs=cfg.save_graphs, save_graphs_path=cfg.save_graphs_path)
    if not isinstance(cfg.device_id, int):
        ms.set_context(device_id=cfg.device_id)
    cfg.need_boost = hasattr(cfg, "boost_level") and cfg.boost_level in ["O1", "O2"]
    set_graph_kernel_context(cfg.device_target, cfg.model_name)
    if cfg.device_num > 1:
        init()
        print("run distribute!", flush=True)
        group_size = get_group_size()
        if cfg.device_num != group_size:
            raise ValueError(f"the setting device_num: {cfg.device_num} not equal to the real group_size: {group_size}")
        cfg.rank_id = get_rank()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        if hasattr(cfg, "all_reduce_fusion_config") and (not cfg.need_boost):
            ms.set_auto_parallel_context(all_reduce_fusion_config=cfg.all_reduce_fusion_config)
        if cfg.device_target == "Ascend":
            set_algo_parameters(elementwise_op_strategy_follow=True)
    else:
        cfg.device_num = 1
        cfg.rank_id = 0
        print("run standalone!", flush=True)
