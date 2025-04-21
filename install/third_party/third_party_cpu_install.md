# 源码编译方式安装MindSpore CPU版本（含第三方依赖）

[![查看源文件](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.6.0rc1/resource/_static/logo_source.svg)](https://gitee.com/mindspore/docs/blob/r2.6.0rc1/install/third_party/third_party_cpu_install.md)

作者：[damon0626](https://gitee.com/damon0626)

本文档介绍如何在```Ubuntu 18.04 64```位操作系统```CPU```环境下，使用源码编译方式安装```MindSpore```。

## 确认系统环境信息

### 1. 确认安装Ubuntu 18.04是64位操作系统

（1）确认系统版本号，在终端输入```lsb_release -a```

```text
ms-sd@mssd:~$ lsb_release -a
No LSB modules are available.
Distributor ID:Ubuntu
Description:Ubuntu 18.04.5 LTS
Release:18.04
Codename:bionic
```

（2）确认系统位数，在终端输入```uname -a```

```text
ms-sd@mssd:~$ uname -a
Linux mssd 5.4.0-42-generic #46~18.04.1-Ubuntu SMP Fri Jul 10 07:21:24 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux
```

### 2. 确认安装GCC 7.3.0版本

（1）确认当前系统安装的GCC版本

在终端输入```gcc --version```，系统已安装版本为7.5.0

```text
ms-sd@mssd:~/gcc-7.3.0/build$ gcc --version
gcc (Ubuntu 7.5.0-3ubuntu1~18.04) 7.5.0
Copyright (C) 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  
There is NOwarranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

（2）如果提示找不到gcc命令，用以下方式安装

```text
ms-sd@mssd:~$ sudo apt-get install gcc
```

（3）本地编译安装7.3.0，下载文件

[点此下载GCC7.3.0](https://ftp.gnu.org/gnu/gcc/gcc-7.3.0/gcc-7.3.0.tar.gz)

（4）解压并进入目录

```bash
tar -xvzf gcc-7.3.0.tar.gz
cd gcc-7.3.0
```

（5）运行```download_prerequesites```，运行该文件的目的是

> 1. Download some prerequisites needed by gcc.
> 2. Run this from the top level of the gcc source tree and the gcc build will do the right thing.

```text
ms-sd@mssd:~/gcc-7.3.0$ ./contrib/download_prerequisites
2020-12-19 09:58:33 URL: ftp://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.1.0.tar.bz2 [2383840] -> "./gmp-6.1.0.tar.bz2" [1]
2020-12-19 10:00:01 URL: ftp://gcc.gnu.org/pub/gcc/infrastructure/mpfr-3.1.4.tar.bz2 [1279284] -> "./mpfr-3.1.4.tar.bz2" [1]
2020-12-19 10:00:50 URL: ftp://gcc.gnu.org/pub/gcc/infrastructure/mpc-1.0.3.tar.gz [669925] -> "./mpc-1.0.3.tar.gz" [1]
2020-12-19 10:03:10 URL: ftp://gcc.gnu.org/pub/gcc/infrastructure/isl-0.16.1.tar.bz2 [1626446] -> "./isl-0.16.1.tar.bz2" [1]
gmp-6.1.0.tar.bz2: 成功
mpfr-3.1.4.tar.bz2: 成功
mpc-1.0.3.tar.gz: 成功
isl-0.16.1.tar.bz2: 成功
All prerequisites downloaded successfully.
```

（6）运行成功后，进行配置

```text
ms-sd@mssd:~/gcc-7.3.0/build$ ../configure --enable-checking=release --enable-languages=c,c++ --disable-multilib
```

> 参数解释：  
> –enable-checking=release  增加一些检查  
> –enable-languages=c,c++ 需要gcc支持的编程语言  
> –disable-multilib 取消多目标库编译(取消32位库编译)  

（7）编译，根据CPU性能，选择合适的线程数

```text
ms-sd@mssd:~/gcc-7.3.0/build$ make -j 6
```

（8）安装

```text
ms-sd@mssd:~$ sudo make install -j 6
```

（9）验证，看到版本已经变更为7.3.0，安装成功。

```text
ms-sd@mssd:~$ gcc --version
gcc (GCC) 7.3.0
Copyright © 2017 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  
There is NOwarranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### 3. 确认安装Python 3.7.5版本

**注意:** ```Ubuntu 18.04``` 系统自带的 ```python3```版本为```python3.6.9```，系统自带```python```不要删除，防止依赖错误。```Linux```发行版中, ```Debian```系的提供了```update-alternatives```工具，用于在多个同功能的软件，或软件的多个不同版本间选择，这里采用```update-alternatives```工具控制多个Python版本。

（1）查看系统Python版本

```text
ms-sd@mssd:~$ python3 --version
Python3.6.9
```

（2）[点此下载Python 3.7.5安装包](https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz)

（3）解压并进入目录

```text
ms-sd@mssd:~$ tar -xvzf Python-3.7.5.tgz
ms-sd@mssd:~$ cd Python-3.7.5/
```

（4）配置文件路径

```text
ms-sd@mssd:~/Python-3.7.5$ ./configure --prefix=/usr/local/python3.7.5 --with-ssl
```

> 参数解释：  
> --prefix=/usr/local/python3.7.5  
> 可执行文件放在/usr/local/python3.7.5/bin下，  
> 库文件放在/usr/local/python3.7.5/lib，  
> 配置文件放在/usr/local/python3.7.1/include，  
> 其他资源文件放在/usr/local/python3.7.5下  
>  
> --with-ssl：确保pip安装库时能找到SSL

（5）安装必要的依赖

```text
ms-sd@mssd:~/Python-3.7.5$ sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python3-openssl --no-check-certificate
```

（6）编译安装

```text
ms-sd@mssd:~/Python-3.7.5$ make -j 6
ms-sd@mssd:~/Python-3.7.5$ sudo make install -j 6
```

（7）查看当前系统python/python3的指向

```text
ms-sd@mssd:~$ ls -l /usr/bin/ | grep python
lrwxrwxrwx 1 root root          23 10月  8 20:12 pdb3.6 -> ../lib/python3.6/pdb.py
lrwxrwxrwx 1 root root          31 12月 18 21:44 py3versions -> ../share/python3/py3versions.py
lrwxrwxrwx 1 root root           9 12月 18 21:44 python3 -> python3.6
-rwxr-xr-x 2 root root     4526456 10月  8 20:12 python3.6
-rwxr-xr-x 2 root root     4526456 10月  8 20:12 python3.6m
lrwxrwxrwx 1 root root          10 12月 18 21:44 python3m -> python3.6m（）
```

（8）备份原来的python3链接，重新建立新的python3指向以更改python3默认指向

```text
ms-sd@mssd:~/Python-3.7.5$ sudo mv /usr/bin/python3 /usr/bin/python3.bak
ms-sd@mssd:~/Python-3.7.5$ sudo ln -s /usr/local/python3.7.5/bin/python3.7 /usr/bin/python3
```

（9）重新建立pip3指向

```text
ms-sd@mssd:~/Python-3.7.5$ sudo ln -s /usr/local/python3.7.5/bin/pip3 /usr/bin/pip3
```

（10）输入验证，Python已更改为3.7.5版本

```text
ms-sd@mssd:~/Python-3.7.5$ python3
Python 3.7.5 (default, Dec 19 2020, 11:29:09)
[GCC 7.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

（11）更新```update-alternatives```python列表

```bash
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 100
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 150
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 110
```

（12）设置Python默认选项，选择2，默认优先级最高的选项

```text
ms-sd@mssd:~$ sudo update-alternatives --config python
There are 3 choices for the alternative python (providing /usr/bin/python).

  Selection    Path                Priority   Status
------------------------------------------------------------
  0            /usr/bin/python3     150       auto mode
  1            /usr/bin/python2     100       manual mode
* 2            /usr/bin/python3     150       manual mode
  3            /usr/bin/python3.6   110       manual mode

Press <enter> to keep the current choice[*], or type selection number:
```

### 4. 确认安装OpenSSL 1.1.1及以上版本

（1）Ubuntu 18.04自带了OpenSSL 1.1.1

```text
ms-sd@mssd:~/Python-3.7.5$ openssl version
OpenSSL 1.1.1  11 Sep 2018
```

（2）本地编译安装请参考[Ubuntu 18.04 安装新版本openssl](https://www.cnblogs.com/thechosenone95/p/10603110.html)

### 5. 确认安装CMake 3.18.3及以上版本

（1）[点此下载CMake 3.18.5](https://github.com/Kitware/CMake/releases/download/v3.18.5/cmake-3.18.5.tar.gz)

（2）解压并进入文件目录

```text
ms-sd@mssd:~$ tar -zxvf cmake-3.18.5.tar.gz
ms-sd@mssd:~$ cd cmake-3.18.5/
```

（3）编译安装

在源码的README.rst中看到如下文字：

> For example, if you simply want to build and install CMake from source,
> you can build directly in the source tree::
>
> $ ./bootstrap && make && sudo make install
>
> Or, if you plan to develop CMake or otherwise run the test suite, create
> a separate build tree::
>
> $.mkdir cmake-build && cd cmake-build
>
> $./cmake-source/bootsrap && make

选择从源码编译安装，根据提示在终端依次输入以下命令：

```text
ms-sd@mssd:~/cmake-3.18.5$ ./bootstrap
ms-sd@mssd:~/cmake-3.18.5$ make -j 6
ms-sd@mssd:~/cmake-3.18.5$ sudo make install -j 6
```

（4）验证，安装成功

```text
ms-sd@mssd:~$ cmake --version
cmake version 3.18.5

CMake suite maintained and supported by Kitware (kitware.com/cmake).
```

### 6. 确认安装wheel 0.32.0及以上版本

（1）更新pip源

修改 ~/.pip/pip.conf (如果没有该文件，创建一个)， 内容如下：

```bash
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

（2）安装wheel 0.32.0

```shell
ms-sd@mssd:~$ sudo pip3 install wheel==0.32.0
```

（3）查看安装情况

```text
ms-sd@mssd:~$ pip3 list
Package    Version
---------- -------
numpy      1.19.4
pip        20.3.3
setuptools 41.2.0
wheel      0.32.0
```

### 7. 确认安装patch 2.5及以上版本

（1）查看patch版本，ubuntu18.04自带了2.7.6版本

```text
ms-sd@mssd:~$ patch --version
GNU patch 2.7.6
Copyright (C) 2003, 2009-2012 Free Software Foundation, Inc.
Copyright (C) 1988 Larry Wall

License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>.
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Written by Larry Wall and Paul Eggert
```

### 8. 确认安装NUMA 2.0.11及以上版本

（1）如果未安装，使用如下命令下载安装：

```text
ms-sd@mssd:~$ apt-get install libnuma-dev
```

### 9. 确认安装git工具

```text
ms-sd@mssd:~$ sudo apt-get install git
```

## MindSpore源码安装

### 10. 下载MindSpore源码

（1）从代码仓库下载源码

```text
ms-sd@mssd:~$ git clone https://gitee.com/mindspore/mindspore.git
```

（2）安装依赖（根据编译过程中报错，整理如下）

```text
ms-sd@mssd:~$ sudo apt-get install python3.7-dev pybind11  python3-wheel python3-setuptools python3.7-minimal
```

（3）编译（内存占用太大，总是超内存线程被杀死，建议4G以上）

```text
ms-sd@mssd:~/mindspore$ sudo bash build.sh -e cpu -j 2
```

（4）编译成功

大约需要1小时，编译成功，出现如下提示：

```text
CPack: - package: /home/ms-sd/mindspore/build/mindspore/mindspore generated.
success building mindspore project!
---------------- mindspore: build end   ----------------
```

同时在```/mindspore/output/```文件夹下生成了```mindspore-1.1.0-cp37-cp37m-linux_x86_64.whl```文件。

（5）pip3安装MindSpore安装文件

```text
ms-sd@mssd:~/mindspore$ sudo pip3 install /mindspore/output/mindspore-1.1.0-cp37-cp37m-linux_x86_64.whl
```

（6）验证安装是否成功

```bash
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"
```

如果输出：

```text
mindspore version: 版本号
The result of multiplication calculation is correct, MindSpore has been installed successfully!
```

说明MindSpore安装成功了。
