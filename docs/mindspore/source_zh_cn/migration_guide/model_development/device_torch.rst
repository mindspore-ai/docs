.. code-block::

    import os
    import torch
    from torch import nn

    # 如果 GPU 可用，则绑定到 GPU 0，否则绑定到 CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 单 GPU 或者 CPU
    # 将模型部署到指定的硬件
    model.to(device)
    # 将数据部署到指定的硬件
    data.to(device)

    # 在多个 GPU 上分发训练
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0,1,2])
        model.to(device)

        # 设置可用设备
        os.environ['CUDA_VISIBLE_DEVICE']='1'
        model.cuda()