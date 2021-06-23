# 后端编译

`Linux` `Windows` `Ascend` `GPU` `CPU` `环境准备` `初级` `中级`

<a href="https://gitee.com/mindspore/docs/blob/master/docs/faq/source_zh_cn/backend_compile.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/master/resource/_static/logo_source.png"></a>

<font size=3>**Q：MindSpore现支持直接读取哪些其他框架的模型和哪些格式呢？比如PyTorch下训练得到的pth模型可以加载到MindSpore框架下使用吗？**</font>

A： MindSpore采用protbuf存储训练参数，无法直接读取其他框架的模型。对于模型文件本质保存的就是参数和对应的值，可以用其他框架的API将参数读取出来之后，拿到参数的键值对，然后再加载到MindSpore中使用。比如想用其他框架训练好的ckpt文件，可以先把参数读取出来，再调用MindSpore的`save_checkpoint`接口，就可以保存成MindSpore可以读取的ckpt文件格式了。

<br/>

<font size=3>**Q：在使用ckpt或导出模型的过程中，报Protobuf内存限制错误，如何处理？**</font>

A：当单条Protobuf数据过大时，因为Protobuf自身对数据流大小的限制，会报出内存限制的错误。这时可通过设置环境变量`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`解除限制。

<br/>

<font size=3>**Q:PyNative模式和Graph模式的区别？**</font>

A: 在网络执行方面，两个模式使用的算子是一致的，因此相同的网络和算子，分别在两个模式下执行时，精度效果是一致的。由于Graph模式运用了图优化、计算图整图下沉等技术，Graph模式执行网络的性能和效率更高；

在场景使用方面，Graph模式需要一开始就构建好网络结构，然后框架做整图优化和执行，比较适合网络固定没有变化，且需要高性能的场景；

在不同硬件（`Ascend`、`GPU`和`CPU`）资源上都支持这两种模式；

代码调试方面，由于PyNative模式是逐行执行算子，用户可以直接调试Python代码，在代码中任意位置打断点查看对应算子`/api`的输出或执行结果。而Graph模式由于在构造函数里只是完成网络构造，实际没有执行，因此在`construct`函数里打断点无法获取对应算子的输出，只能先指定算子进行打印，然后在网络执行完成后查看输出结果。

<br/>

<font size=3>**Q：请问`c_transforms`和`py_transforms`有什么区别，比较推荐使用哪个？**</font>

A：推荐使用`c_transforms`，因为纯C层执行，所以性能会更好。

原理:`c_transform`底层使用的是C版本`opencv/jpeg-turbo`进行的数据处理，`py_transform`使用的是Python版本的`Pillow`进行数据处理。

<br/>

<font size=3>**Q：缓存服务器异常关闭如何处理？**</font>

A：缓存服务器使用过程中，会进行IPC共享内存和socket文件等系统资源的分配。若允许溢出，在磁盘空间还会存在溢出的数据文件。一般情况下，如果通过`cache_admin --stop`命令正常关闭服务器，这些资源将会被自动清理。

但如果缓存服务器被异常关闭，例如缓存服务进程被杀等，用户需要首先尝试重新启动服务器，若启动失败，则应该依照以下步骤手动清理系统资源：

- 删除IPC资源。

    1. 检查是否有IPC共享内存残留。

        一般情况下，系统会为缓存服务分配4GB的共享内存。通过以下命令可以查看系统中的共享内存块使用情况。

        ```text
        $ ipcs -m
        ------ Shared Memory Segments --------
        key        shmid      owner      perms      bytes      nattch     status
        0x61020024 15532037   root       666        4294967296 1
        ```

        其中，`shmid`为共享内存块id，`bytes`为共享内存块的大小，`nattch`为链接到该共享内存块的进程数量。`nattch`不为0表示仍有进程使用该共享内存块。在删除共享内存前，需要停止使用该内存块的所有进程。

    2. 删除IPC共享内存。

        找到对应的共享内存id，并通过以下命令删除。

        ```text
        ipcrm -m {shmid}
        ```

- 删除socket文件。

    一般情况下，socket文件位于`/tmp/mindspore/cache`。进入文件夹，执行以下命令删除socket文件。

    ```text
    rm cache_server_p{port_number}
    ```

    其中`port_number`为用户创建缓存服务器时指定的端口号，默认为50052。

- 删除溢出到磁盘空间的数据文件。

    进入启用缓存服务器时指定的溢出数据路径。通常，默认溢出路径为`/tmp/mindspore/cache`。找到路径下对应的数据文件夹并逐一删除。

<br/>

<font size=3>**Q：编译应用时报错`bash -p`方式和 `bash -e`方式的区别？**</font>

A：MindSpore Serving的编译和运行依赖MindSpore，Serving提供两种编译方式：一种指定已安装的MindSpore路径，即`bash -p {python site-packages}/mindspore/lib`，避免编译Serving时再编译MindSpore；另一种，编译Serving时，编译配套的MindSpore，Serving会将`-e`、`-V`和`-j`选项透传给MindSpore。
比如，在Serving目录下，`bash -e ascend -V 910 -j32`：

- 首先将会以`bash -e ascend -V 910 -j32`方式编译`third_party/mindspore`目录下的MindSpore；
- 其次，编译脚本将MindSpore编译结果作为Serving的编译依赖。

<br/>

<font size=3>**Q：运行应用时报错`libmindspore.so: cannot open shared object file: No such file or directory`怎么办？**</font>

A：首先，需要确认是否安装MindSpore Serving所依赖的MindSpore；其次，Serving 1.1需要配置`LD_LIBRARY_PATH`，显式指定`libmindspore.so`所在路径，`libmindspore.so`当前在MindSpore Python安装路径的`lib`目录下；Serving 1.2后不再需要显示指定`libmindspore.so`所在路径，Serving会基于MindSpore安装路径查找并追加配置`LD_LIBRARY_PATH`，用户不再需要感知。
