.. code-block::

    import mindspore as ms
    ms.set_context(device_target='Ascend', device_id=0)

    # define net
    Model = ..
    # define dataset
    dataset = ..
    # training, automatically deploy to Ascend according to device_target
    Model.train(1, dataset)