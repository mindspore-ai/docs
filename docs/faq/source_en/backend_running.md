# Backend Running

`Ascend` `GPU` `CPU` `Environmental Setup` `Operation Mode` `Model Training` `Beginner` `Intermediate` `Expert`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_en/backend_running.md" target="_blank"><img src="./_static/logo_source.png"></a>

Q: How does MindSpore implement the early stopping function?

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

Q: What can I do if an error message `wrong shape of image` is displayed when I use a model trained by MindSpore to perform prediction on a `28 x 28` digital image with white text on a black background?

A: The MNIST gray scale image dataset is used for MindSpore training. Therefore, when the model is used, the data must be set to a `28 x 28` gray scale image, that is, a single channel.

<br/>

Q: For Ascend users, how to get more detailed logs when the `run task error` is reported?

A: More detailed logs info can be obtained by modify slog config file. You can get different level by modify `/var/log/npu/conf/slog/slog.conf`. The values are as follows: 0:debug、1:info、2:warning、3:error、4:null(no output log), default 1.

<br/>

Q: What can I do if the error message `Pynative run op ExpandDims failed` is displayed when the ExpandDims operator is used? The code is as follows:

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

You can select a proper mode and writing method to complete the training by referring to the official website [tutorial](https://www.mindspore.cn/tutorial/training/en/master/advanced_use/debug_in_pynative_mode.html).

<br/>

Q: How to fix the error below when running MindSpore distributed training with GPU:

```text
Loading libgpu_collective.so failed. Many reasons could cause this:
1.libgpu_collective.so is not installed.
2.nccl is not installed or found.
3.mpi is not installed or found
```

A: This message means that MindSpore failed to load library `libgpu_collective.so`. The Possible causes are:

- OpenMPI or NCCL is not installed in this environment.
- NCCL version is not updated to `v2.7.6`: MindSpore `v1.1.0` supports GPU P2P communication operator which relies on NCCL `v2.7.6`. `libgpu_collective.so` can't be loaded successfully if NCCL is not updated to this version.
