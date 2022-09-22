# FAQ

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindpandas/docs/source_zh_cn/faq.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

<font size=3>**Q: 请问yrctl start命令执行后未能成功部署计算引擎，需要检查哪些内容？**</font>

A: --address参数指定的地址是否被代理转发。可以通过echo $http_proxy查看系统是否设置了http代理。需要取消代理或将ip地址加入no_proxy变量。

查看系统中是否已经有其它redis服务正在运行导致端口冲突。mindpandas的redis默认运行在6379端口，如需修改，需要在mindpandas的安装目录下修改mindpandas/dist_executor/modules/config/config.xml中的redis_port字段为其它不冲突的端口。

尝试先执行yrctl stop --master，清理残留的服务，之后再运行yrctl start。
