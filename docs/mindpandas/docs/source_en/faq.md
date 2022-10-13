# FAQ

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindpandas/docs/source_en/faq.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

<font size=3>**Q: What should I check when I encounter an error with the message "Error: exit status 1" or "Failed to deploy basic model" when deploying a multi-process compute engine? **</font>

A: There are many possibilities for the engine failure to start. Some common reasons are listed below:

- Reason 1: The IP address of the master node is forwarded by the system's proxy.
   Solution: You can enter `echo $http_proxy` on the command line to check if the system has set an http proxy. If so, you need to cancel the proxy, or add the IP address of the master node to the `$no_proxy` variable.
- Reason 2: Redis port conflict.
   Solution: You can enter `ps -ef|grep redis` on the command line to check whether other redis services are running in the system, resulting in port conflicts. MindPandas' redis runs on port 6379 by default. If you need to modify it, you can modify the `redis_port` field in `mindpandas/dist_executor/modules/config/config.xml` in the MindPandas installation directory to other non-conflicting ports.
- Reason 3: etcd port conflict.
   Solution: You can enter `netstat -tunpl|grep -E "32379|32380"` in the command line to check whether the etcd port is occupied. If there is a conflict, please try to release the corresponding port.

<br/>

<font size=3>**Q: How to solve the error "\*\*ERROR\*\* memory for function instances deployment is less than 0" when deploying a multi-process compute engine? **</font>

A: The problem is caused by insufficient running memory. Please try to decrease the value of the `--datamem` parameter or increase the value of the `--mem` parameter when deploying.

<br/>

<<font size=3>**Q: What should I do if the error message "Failed to request, code:1001, message: invalid resource parameter, request resource is greater than each node's max resource." is reported when running a Python script using a multi-process backend? **</font>

A: This error is caused by insufficient resources configured when starting the distributed compute engine. Please use larger `--cpu` and `--mem` parameter values when deploying the cluster.

<br/>

<font size=3>**Q: When using a multi-process backend, what should I do if "Client number upper to the limit" is reported during the Python script? **</font>

A: Please try to redeploy the cluster and reduce the value of the `--cpu` parameter.

<br/>

<font size=3>**Q: What should I do if the error message "health check failed, please check port: \<port>" is reported during the deployment of a multi-process compute engine? **</font>

A: The MindPandas compute engine will start multiple processes, and each process has a corresponding port. If the ports conflict, this error will be reported. The solutions are as follows:

- To check whether the port that is reporting the error is occupied, you can use the shell command `netstat -tunpl|grep <port>` to check the port occupancy. If the port conflicts, there are two solutions:
     - Method 1: Release the occupation of the conflicting port.
     - Method 2: Modify the port used by the compute engine. In the MindPandas installation directory `dist_executor/modules/config/config.xml`, search for the conflicting port number and change it to another available port.
- If there is no conflict between the ports, you need to check whether the proxy is set. If so, please remove the `$http_proxy` environment variable.
- To check whether there is a residual process from the last startup, you can use `ps -ef |grep mindpandas/dist_executor` to check the PID of the residual process, and then manually clean up the process.

<br/>

<font size=3>**Q: What should I do when the error message "failed to request, code:3003, put object failed, id:\<id>,requestID:\<id>,errr: code:[Out of memory]" is reported during using a multi-process backend? **</font>

A: It may be due to insufficient shared memory space of the compute engine. Please try to stop the engine and then redeploy, and set a larger `--datamem` parameter value.

<br/>

<font size=3>**Q: What should I do when the error message "Failed to request, code:1001, message: invalid resource parameter, request resource is greater than each node's max resource" is reported during using a multi-process backend? **</font>

A: It may be because the CPU and memory resources requested during deployment are too few. Please try the following solutions:

- Configure larger CPU and memory resources when deploying the engine.
- Use a multithreaded backend.

<br/>

<font size=3>**Q: How to solve the "RuntimeError: system not initialized" error when running on a machine with large specifications (such as more than 100 CPU cores)? **</font>

A: Data transfer in compute engine relies on file descriptors. It is required that the number of available file descriptors should be at least four times the number of CPU cores in the cluster. You can view and increase the limit on the number of file descriptors on the current machine through the `ulimit` command:

```shell
$ ulimit –a  # Where open files is the upper limit of the file descriptor. If the value is too small, it will be raised.
open files                      (-n) 1024
$ ulimit -n 4096
```

<br/>

<font size=3>**Q: How to solve "ImportError: /lib/libc.so.6: version \`GLIBC_2.25\` not found" when using multi-process backend? **</font>

A: Please upgrade the glibc version in the environment to 2.25 or above.

<br/>

<font size=3>**Q: How to solve the error "TypeError: cannot unpack non-iterable <class 'yr.exception.YRInvokeError'> object" when I use the `pytest` command to execute a script in a multi-process backend? **</font>

A: Due to the execution mechanism of `pytest`, if you use a user-defined function, please make sure that other functions called in it are Python closures.

<br/>

<font size=3>**Q: How to solve the "yr.exception.YRequestError: failed to request, code:3003, message: retry etcd operation Put exceed the max times" message when running with a multi-process backend? **</font>

A: The compute engine uses etcd to maintain the consistency of internal data. This error may be not working properly caused by etcd. You can use the following command to check whether the etcd process exists. If the process does not exist, you need to redeploy the compute engine.

```shell
ps -ef |grep dist_executor/modules/basic/bin/etcd/etcd
```

<br/>

<font size=3>**Q: How to solve "RuntimeError: code: [RPC unavailable], msg [ Thread ID && RPC unavailable. Disconnected from worker . Line of code : 117 File : object_client_impl.cpp]" when running ? **</font>

A: It may be that the module using rpc communication in the compute engine is abnormal. Please use the following command to check the corresponding process. If the number of processes found is less than 3, the compute engine needs to be redeployed.

```shell
ps -ef |grep dist_executor/modules/datasystem
```

<br/>

<font size=3>**Q: How to solve the "xmllint: command not found" when a multi-process compute engine is deployed? **</font>

A: Install libxml2-utils to solve this problem.
