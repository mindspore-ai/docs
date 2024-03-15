.. code-block::

    import mindspore as ms
    ms.set_context(device_target='Ascend', device_id=0)

    # 定义网络
    Model = ..
    # 定义数据集
    dataset = ..
    # 训练，根据 device_target 自动部署到 Ascend
    Model.train(1, dataset)