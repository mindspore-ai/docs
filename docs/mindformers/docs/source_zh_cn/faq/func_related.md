# 功能相关

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/master/docs/mindformers/docs/source_zh_cn/faq/func_related.md)

## Q: 如何生成模型切分策略文件？

A: 模型切分策略文件记录了模型权重在分布式场景下的切分策略，一般在离线权重切分时使用。在网络`yaml`文件中配置`only_save_strategy: True`，然后正常启动分布式任务，便可在`output/strategy/`目录下生成分布式策略文件，详细介绍请参阅[分布式权重切分与合并教程](https://www.mindspore.cn/mindformers/docs/zh-CN/dev/function/transform_weight.html#%E7%A6%BB%E7%BA%BF%E8%BD%AC%E6%8D%A2%E9%85%8D%E7%BD%AE%E8%AF%B4%E6%98%8E)。

<br/>

## Q: 生成`rank_table`文件报错`socket.gaierror: [Errno -2] Name or service not known`或者`socket.gaierror: [Errno -3] Temporary failure in name resolution`，怎么解决？

A: 主要原因是在`docker`中运行时，需要先获取到主机名。可以通过如下脚本获取主机名, 如`xxxx42`

```python
import socket

def get_host_name():
    try:
        hostname = socket.gethostname()
        print(f"Hostname: {hostname}")
    except EOFError:
        pass

get_host_name()
```

然后将主机名添加到`/etc/hosts`中：

```text
127.0.0.1 xxxx42
```

再运行`hccl_tools.py`就可以了。

<br/>
