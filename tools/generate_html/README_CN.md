# MindSpore文档构建

> 运行之前确认已安装：
>
> - python>=3.7
> - git
> - pandoc
> - doxygen

1. 配置环境变量`work_dir`作为工作目录。

    ```bash
    export work_dir=/home/useranme/xxx`
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

    ```bash
    python run.py --version="1.9.0" --user="" --pd="" --wgetdir=""
    ```

    | 参数 | 值 | 必填 |
    | ---- | ---- | ---- |
    | version | 构建的版本号 | 是 |
    | user | 每日构建网站的用户名 | 否 |
    | pd | 每日构建网站的密码 | 否 |
    | wgetdir | 每日构建网站网址 | 否 |
