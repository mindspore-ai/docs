# FAQ

<a href="https://gitee.com/mindspore/docs/blob/r2.0.0-alpha/docs/mindpandas/docs/source_zh_cn/faq.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

<font size=3>**Q：请问部署多进程计算引擎时遇到“Error: exit status 1”或“Failed to deploy basic model”报错，需要检查哪些内容？**</font>

A：引擎启动失败有多种可能性，下面列举了一些常见的原因，请参照检查。

- 原因1：master节点的IP地址被系统的代理转发了。  
  解决方案：可以在命令行中输入`echo $http_proxy`，查看系统是否设置了http代理。如果有则需要取消代理，或将master节点的IP地址加入`$no_proxy`变量。
- 原因2：redis端口冲突。  
  解决方案：可以在命令行中输入`ps -ef|grep redis`，查看系统中是否已经有其它redis服务正在运行，导致端口冲突。MindPandas的redis默认运行在6379端口，如需修改，可以在MindPandas的安装目录下修改`mindpandas/dist_executor/modules/config/config.xml`中的`redis_port`字段为其它不冲突的端口。
- 原因3：etcd端口冲突。  
  解决方案：可以在命令行中输入`netstat -tunpl|grep -E "32379|32380"`，查看etcd的端口是否已被占用，如果发生冲突，请尝试解除相应端口的占用。

<br/>

<font size=3>**Q：请问部署多进程计算引擎时，报“\*\*ERROR\*\* memory for function instances deployment is less than 0”错误如何解决？**</font>

A：该问题是由于运行内存不足导致的，请尝试在部署时减小`--datamem`参数值或增大`--mem`参数值。

<br/>

<font size=3>**Q：请问使用多进程后端运行Python脚本时报“Failed to request, code:1001, message: invalid resource parameter, request resource is greater than each node's max resource.”该如何解决？**</font>

A：此报错是启动分布式计算引擎时配置的资源不足导致的，请尝试部署集群时使用更大的`--cpu`和`--mem`参数值。

<br/>

<font size=3>**Q：请问使用多进程后端时，运行Python脚本报“Client number upper to the limit”该如何解决？**</font>

A：请尝试重新部署集群并减小`--cpu`参数的值。

<br/>

<font size=3>**Q：在部署多进程计算引擎的过程中出现“health check failed, please check port: \<port>”应如何解决？**</font>

A：MindPandas计算引擎会启动多个进程，每个进程都有对应的端口，若端口冲突则会导致此报错。解决方法如下：

- 查看报错的端口是否被占用，可以通过shell指令`netstat -tunpl|grep <port>`查看端口占用情况，若端口冲突，有两种解决方案：
    - 方法1：解除冲突端口的占用。
    - 方法2：修改计算引擎使用的端口。在MindPandas安装目录下`dist_executor/modules/config/config.xml`里，搜索发生冲突的端口号，将其修改为其他空闲端口。
- 若端口无冲突，需要查看是否设置了代理，如有请移除`$http_proxy`环境变量。
- 查看是否有上次启动残留的进程，可以使用`ps -ef |grep mindpandas/dist_executor`查看残留进程PID，然后手动清理进程。

<br/>

<font size=3>**Q：使用多进程模式在运行的过程中出现报错“failed to request, code:3003, put object failed, id:\<id>,requestID:\<id>,errr:code:[Out of memory]”如何解决？**</font>

A：可能是由于计算引擎的共享内存空间不足，请尝试停止引擎后重新部署，并设置更大的`--datamem`参数值。

<br/>

<font size=3>**Q：在多进程后端运行的过程中出现报错“Failed to request, code:1001, message: invalid resource parameter, request resource is greater than each node’s max resource”该如何解决？**</font>

A：可能是由于部署时申请的CPU和内存资源太少，请尝试下列解决方案：

- 部署引擎时配置更大的CPU和内存资源。
- 使用多线程后端。

<br/>

<font size=3>**Q：在大规格（如CPU核心数大于100）的机器上运行时，报“RuntimeError: system not initialized”错误如何解决？**</font>

A：计算引擎中的数据传输依赖文件描述符。要求系统可用文件描述符的个数应至少为集群CPU核心数的四倍。可以通过`ulimit`指令查看，并提高当前机器的文件描述符个数限制：

```shell
$ ulimit –a  # 其中open files为文件描述符的上限值，若该数值过小，上调
open files                      (-n) 1024
$ ulimit -n 4096
```

<br/>

<font size=3>**Q：使用多进程后端时报“ImportError: /lib/libc.so.6: version \`GLIBC_2.25\` not found”如何解决？**</font>

A：请升级环境中的glibc版本到2.25或以上。

<br/>

<font size=3>**Q：多进程后端下使用`pytest`命令执行脚本，报“TypeError: cannot unpack non-iterable <class ‘yr.exception.YRInvokeError’> object”错误如何解决？**</font>

A：由于`pytest`的执行机制原因，如果您使用了用户自定义函数，请确保其中调用到的其他函数是Python闭包。

<br/>

<font size=3>**Q：使用多进程后端运行时报“yr.exception.YRequestError: failed to request, code:3003, message: retry etcd operation Put exceed the max times”如何解决？**</font>

A：计算引擎使用etcd来维护内部数据的一致性，此报错可能是因为etcd未能正常工作。可以使用下述指令查看etcd进程是否存在，若进程不在，则需要重新部署计算引擎。

```shell
ps -ef |grep dist_executor/modules/basic/bin/etcd/etcd
```

<br/>

<font size=3>**Q：运行的时候报“RuntimeError: code: [RPC unavailable],msg [ Thread ID && RPC unavailable. Disconnected from worker . Line of code :117 File : object_client_impl.cpp]”如何解决？**</font>

A：可能是计算引擎中使用rpc通信的模块发生异常，请使用下述指令检查相应进程，若查找出来的进程数量少于3个，则需要重新部署计算引擎。

```shell
ps -ef |grep dist_executor/modules/datasystem
```

<br/>

<font size=3>**Q：部署多进程计算引擎时报“xmllint:command not found”如何解决？**</font>

A：安装libxml2-utils即可解决此问题。
