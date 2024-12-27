.. code-block::

    import mindspore as ms
    ms.set_device("Ascend", 0)

    # define net
    Model = ..
    # define dataset
    dataset = ..
    # training, automatically deploy to Ascend according to device_target
    Model.train(1, dataset)