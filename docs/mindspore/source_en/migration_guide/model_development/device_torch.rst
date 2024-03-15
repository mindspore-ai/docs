.. code-block::

    import os
    import torch
    from torch import nn

    # bind to the GPU 0 if GPU is available, otherwise bind to CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Single GPU or CPU
    # deploy model to specified hardware
    model.to(device)
    # deploy data to specified hardware
    data.to(device)

    # distribute training on multiple GPUs
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0,1,2])
        model.to(device)

        # set available device
        os.environ['CUDA_VISIBLE_DEVICE']='1'
        model.cuda()