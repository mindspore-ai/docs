# MindSpore文档构建

> 运行之前确认已安装：
>
> - python>=3.7
> - git
> - pandoc
> - doxygen

1. 配置环境变量`work_dir`作为工作目录。

    ```bash
    export work_dir=/home/useranme/xxx
    ```

    默认在当前目录进行构建。

2. 准备好json文件。

    整体结构为列表套字典: `[{},{},{},...]`
    每日构建html取名为: `daily.json`
    其余版本构建html取名为: `version.json`

    如下对json中键值对进行举例：

    ```bash
    "id" : 9 ,
    "name" : "lite",
    "branch" : "r1.9",
    "whl_path" : "/MindSpore/lite/release/linux/x86_64/",
    "whl_name" : "mindspore_lite-1.9.0-cp37-cp37m-linux_x86_64.whl",
    "environ" : "MS_PATH",
    "uninstall_name" : "mindspore_lite",
    "tar_path" : "/MindSpore/lite/release/linux/x86_64/",
    "tar_name" : "mindspore-lite-1.9.0-linux-x64.tar.gz"
    ```

    | 键 | 值 | 必填 |
    | ---- | ---- | ---- |
    | id | 计数用 | 否 |
    | name | 组件的名称 | 是 |
    | branch | 组件的分支 | 是 |
    | whl_path | 组件whl安装包的文件夹路径 | 是 |
    | whl_name | 组件whl安装包的名称 | 是 |
    | environ | 组件仓库需要配置的环境变量名 | 是 |
    | uninstall_name | 组件whl安装包的卸载命令名称 | 是 |
    | tar_path | 组件tar包的文件夹路径 | 否 |
    | tar_name | 组件tar包的名称 | 否 |

    必填键，若无对应值，其值填入空。

3. 运行：

    比如构建1.9.0的发布版本：

    ```bash
    python run.py --version="1.9.0" --user="" --pd="" --wgetdir="" --release_url="https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.9.0" --theme="xxx/xxx"
    ```

    支持单独构建某几个组件。（该功能从r2.2分支开始支持）
    如构建2.2.0的发布版本的 ``mindspore`` , ``sciai`` 组件：

    ```bash
    python run.py --version="2.2.0" --user="" --pd="" --wgetdir="" --release_url="https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.0" --theme="xxx/xxx" --single_generate="mindspore,sciai"
    ```

    | 参数 | 值 | 必填 | 适配分支 |
    | ---- | ---- | ---- | ---- |
    | version | 构建的版本号 | 是 | 全部 |
    | theme | theme样式的文件夹路径 | 是 | >= r2.0.0-alpha |
    | release_url | 构建发布版本的基础网址 | 否 | 全部 |
    | user | 每日构建网站的用户名 | 否 | 全部 |
    | pd | 每日构建网站的密码 | 否 | 全部 |
    | wgetdir | 每日构建网站网址 | 否 | 全部 |
    | single_generate | 单独构建某个组件 | 否 | >= r2.2 |
