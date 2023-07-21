# Backend Running

`Ascend` `GPU` `CPU` `Environmental Setup` `Operation Mode` `Model Training` `Beginner` `Intermediate` `Expert`

[![View Source On Gitee](./_static/logo_source.png)](https://gitee.com/mindspore/docs/blob/r1.1/docs/faq/source_en/backend_running.md)

<font size=3>**Q: How do I view the number of model parameters?**</font>

A: You can load the checkpoint to count the parameter number. Variables in the momentum and optimizer may be counted, so you need to filter them out.
You can refer to the following APIs to collect the number of network parameters:

```python
def count_params(net):
    """Count number of parameters in the network
    Args:
        net (mindspore.nn.Cell): Mindspore network instance
    Returns:
        total_params (int): Total number of trainable params
    """
    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)
    return total_params
```

[Script Link](https://gitee.com/mindspore/mindspore/blob/r1.1/model_zoo/research/cv/tinynet/src/utils.py).

<br/>

<font size=3>**Q: How do I build a multi-label MindRecord dataset for images?**</font>

A: The data schema can be defined as follows:`cv_schema_json = {"label": {"type": "int32", "shape": [-1]}, "data": {"type": "bytes"}}`

Note: A label is an array of the numpy type, where label values 1, 1, 0, 1, 0, 1 are stored. These label values correspond to the same data, that is, the binary value of the same image.
For details, see [Converting Dataset to MindRecord](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/convert_dataset.html#id3).

<br/>

<font size=3>**Q: How do I monitor the loss during training and save the training parameters when the `loss` is the lowest?**</font>

A: You can customize a `callback`.For details, see the writing method of `ModelCheckpoint`. In addition, the logic for determining loss is added.

```python
class EarlyStop(Callback):
def __init__(self):
    self.loss = None
def step_end(self, run_context):
     loss =  ****(get current loss)
     if (self.loss == None or loss < self.loss):
         self.loss = loss
         # do save ckpt
```

<br/>

<font size=3>**Q: How do I execute a single `ut` case in `mindspore/tests`?**</font>

A: `ut` cases are usually based on the MindSpore package of the debug version, which is not provided on the official website. You can run `sh build.sh` to compile the source code and then run the `pytest` command. The compilation in debug mode does not depend on the backend. Run the `sh build.sh -t on` command. For details about how to execute cases, see the `tests/runtest.sh` script.

<br/>

<font size=3>**Q: How do I obtain the expected `feature map` when `nn.Conv2d` is used?**</font>

A: For details about how to derive the `Conv2d shape`, click [here](https://www.mindspore.cn/doc/api_python/en/r1.1/mindspore/nn/mindspore.nn.Conv2d.html#mindspore.nn.Conv2d.) Change `pad_mode` of `Conv2d` to `same`. Alternatively, you can calculate the `pad` based on the Conv2d shape derivation formula to keep the `shape` unchanged. Generally, the pad is `(kernel_size-1)//2`.

<br/>

<font size=3>**Q: What can I do if the network performance is abnormal and weight initialization takes a long time during training after MindSpore is installed?**</font>

A: The `SciPy 1.4` series versions may be used in the environment. Run the `pip list | grep scipy` command to view the `SciPy` version and change the `SciPy` version to that required by MindSpore. You can view the third-party library dependency in the `requirement.txt` file.
<https://gitee.com/mindspore/mindspore/blob/{version}/requirements.txt>
> Replace version with the specific version branch of MindSpore.

<br/>

<font size=3>**Q: Can MindSpore be used to customize a loss function that can return multiple values?**</font>

A: After customizing the `loss function`, you need to customize `TrainOneStepCell`. The number of `sens` for implementing gradient calculation is the same as the number of `network` outputs. For details, see the following:

```python
net = Net()

loss_fn = MyLoss()

loss_with_net = MyWithLossCell(net, loss_fn)

train_net = MyTrainOneStepCell(loss_with_net, optim)

model = Model(net=train_net, loss_fn=None, optimizer=None)
```

<br/>

<font size=3>**Q: How does MindSpore implement the early stopping function?**</font>

A: You can customize the `callback` method to implement the early stopping function.
Example: When the loss value decreases to a certain value, the training stops.

```python
class EarlyStop(Callback):
    def __init__(self, control_loss=1):
        super(EarlyStep, self).__init__()
        self._control_loss = control_loss

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        if loss.asnumpy() < self._control_loss:
            # Stop training.
            run_context._stop_requested = True

stop_cb = EarlyStop(control_loss=1)
model.train(epoch_size, ds_train, callbacks=[stop_cb])
```

<br/>

<font size=3>**Q: What can I do if an error message `wrong shape of image` is displayed when I use a model trained by MindSpore to perform prediction on a `28 x 28` digital image with white text on a black background?**</font>

A: The MNIST gray scale image dataset is used for MindSpore training. Therefore, when the model is used, the data must be set to a `28 x 28` gray scale image, that is, a single channel.

<br/>

<font size=3>**Q: What can I do if the error message `device target [CPU] is not supported in pynative mode` is displayed for the operation operator of MindSpore?**</font>

A: Currently, the PyNative mode supports only Ascend and GPU and does not support the CPU.

<br/>

<font size=3>**Q: For Ascend users, how to get more detailed logs when the `run task error` is reported?**</font>

A: Use the msnpureport tool to set the on-device log level. The tool is stored in `/usr/local/Ascend/driver/tools/msnpureport`.

```bash
- Global: /usr/local/Ascend/driver/tools/msnpureport -g info
```

```bash
- Module-level: /usr/local/Ascend/driver/tools/msnpureport -m SLOG:error
```

```bash
- Event-level: /usr/local/Ascend/driver/tools/msnpureport -e disable/enable
```

```bash
- Multi-device ID-level: /usr/local/Ascend/driver/tools/msnpureport -d 1 -g warning
```

Assume that the value range of deviceID is [0, 7], and `devices 0–3` and `devices 4–7` are on the same OS. `Devices 0–3` share the same log configuration file and `devices 4–7` share the same configuration file. In this way, changing the log level of any device (for example device 0) will change that of other devices (for example `devices 1–3`). This rule also applies to `devices 4–7`.

After the driver package is installed (assuming that the installation path is /usr/local/HiAI and the execution file `msnpureport.exe` is in the C:\ProgramFiles\Huawei\Ascend\Driver\tools\ directory on Windows), run the command in the /home/shihangbo/ directory to export logs on the device to the current directory and store logs in a folder named after the timestamp.

<br/>

<font size=3>**Q: What can I do if the error message `Pynative run op ExpandDims failed` is displayed when the ExpandDims operator is used? The code is as follows:**</font>

```python
context.set_context(
mode=cintext.GRAPH_MODE,
device_target='ascend')
input_tensor=Tensor(np.array([[2,2],[2,2]]),mindspore.float32)
expand_dims=ops.ExpandDims()
output=expand_dims(input_tensor,0)
```

A: The problem is that the Graph mode is selected but the PyNative mode is used. As a result, an error is reported. MindSpore supports the following running modes which are optimized in terms of debugging or running:

- PyNative mode: dynamic graph mode. In this mode, operators in the neural network are delivered and executed one by one, facilitating the compilation and debugging of the neural network model.
- Graph mode: static graph mode. In this mode, the neural network model is compiled into an entire graph and then delivered for execution. This mode uses technologies such as graph optimization to improve the running performance and facilitates large-scale deployment and cross-platform running.

You can select a proper mode and writing method to complete the training by referring to the official website [tutorial](https://www.mindspore.cn/tutorial/training/en/r1.1/advanced_use/debug_in_pynative_mode.html).

<br/>

<font size=3>**Q: How to fix the error below when running MindSpore distributed training with GPU:**</font>

```text
Loading libgpu_collective.so failed. Many reasons could cause this:
1.libgpu_collective.so is not installed.
2.nccl is not installed or found.
3.mpi is not installed or found
```

A: This message means that MindSpore failed to load library `libgpu_collective.so`. The Possible causes are:

- OpenMPI or NCCL is not installed in this environment.
- NCCL version is not updated to `v2.7.6`: MindSpore `v1.1.0` supports GPU P2P communication operator which relies on NCCL `v2.7.6`. `libgpu_collective.so` can't be loaded successfully if NCCL is not updated to this version.
