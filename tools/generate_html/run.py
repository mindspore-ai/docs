"""
使用json文件自动化生成mindspore各组件的html页面
"""
import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import time
import requests
import sphinx
import urllib3
from git import Repo
from lxml import etree


# 下载仓库
def git_clone(repo_url, repo_dir):
    if not os.path.exists(repo_dir):
        print("Cloning repo.....")
        os.makedirs(repo_dir, exist_ok=True)
        Repo.clone_from(repo_url, repo_dir, branch='master')
        print("Cloning Repo Done.")

# 更新仓库
def git_update(repo_dir, branch):
    repo = Repo(repo_dir)
    str1 = repo.git.execute(["git", "clean", "-dfx"])
    print(str1)
    str2 = repo.git.execute(["git", "reset", "--hard", "HEAD"])
    print(str2)
    str3 = repo.git.execute(["git", "checkout", branch])
    print(str3)
    str4 = repo.git.execute(["git", "pull", "origin", branch])
    print(str4)

pythonlib_dir = os.path.dirname(os.path.dirname(sphinx.__file__))

def deal_err(err):
    extra_str_re = re.compile(r"\[3.*?m")
    workdir_re = re.compile(rf"{REPODIR}")
    pythonlib_re = re.compile(rf"{pythonlib_dir}")
    err_new = extra_str_re.sub('', err)
    err_new = workdir_re.sub('', err_new)
    err_new = pythonlib_re.sub('', err_new)
    return err_new

def flush(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

#######################################
# 运行检测
#######################################
def main(version, user, pd, WGETDIR):

    print(f"开始构建{version}版本html....")

    # 保存需要生成html的文件夹及对应版本号
    ArraySource = dict()

    # 不同版本保存路径
    WORKDIR = f"{MAINDIR}/{version}"

    # html文档页面保存路径
    OUTPUTDIR = f"{WORKDIR}/output"

    # 各个组件安装包下载保存路径
    WHLDIR = f"{WORKDIR}/whlpkgs"

    # 开始计时
    time_start = time.perf_counter()

    # 读取json文件数据
    if version == "daily":
        with open(os.path.join(os.path.dirname(__file__), "daily.json"), 'r+', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(os.path.join(os.path.dirname(__file__), "version.json"), 'r+', encoding='utf-8') as f:
            data = json.load(f)

    flush(WHLDIR)
    # 遍历json数据做好生成html前的准备
    # pylint: disable=R1702
    for i in range(len(data)):
        # 特殊与一般性的往ArraySource中加入键值对
        if data[i]['name'] == "lite":
            ArraySource[data[i]['name'] + '/docs'] = data[i]["branch"]
            ArraySource[data[i]['name'] + '/api'] = data[i]["branch"]
            ArraySource[data[i]['name'] + '/faq'] = data[i]["branch"]
        elif data[i]['name'] == "tutorials":
            ArraySource[data[i]['name']] = data[i]["branch"]
            ArraySource[data[i]['name'] + '/application'] = data[i]["branch"]
            ArraySource[data[i]['name'] + '/experts'] = data[i]["branch"]
        elif data[i]['name'] == "mindspore":
            ArraySource[data[i]['name']] = data[i]["branch"]
        else:
            ArraySource[data[i]['name'] + '/docs'] = data[i]["branch"]

        # 克隆仓库与配置环境变量
        repo_name = data[i]['name'].replace('_', '-')
        repo_url = f"https://gitee.com/mindspore/{repo_name}.git"
        repo_path = f"{REPODIR}/{data[i]['name']}"
        branch_ = data[i]["branch"]
        if  data[i]['name'] == "devtoolkit":
            repo_url = f"https://gitee.com/mindspore/ide-plugin.git"
            repo_path = f"{REPODIR}/ide-plugin"

        status_code = requests.get(f"{repo_url}").status_code
        if status_code == 200:
            try:
                git_clone(repo_url, repo_path)
                git_update(repo_path, branch_)
                if data[i]['environ']:
                    os.environ[data[i]['environ']] = repo_path
            except KeyError:
                print(f'{repo_name}仓库克隆或更新失败')
        else:
            print(f'{repo_name}对应git仓库访问错误，跳过克隆阶段。。。')

        # 卸载原来已有的安装包, 以防冲突
        if data[i]['uninstall_name']:
            cmd_uninstall = ["pip", "uninstall", "-y", f"{data[i]['uninstall_name']}"]
            subprocess.run(cmd_uninstall)

        os.chdir(WHLDIR)

        # 从网站下载各个组件需要的whl包或tar包
        if version == "daily":
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            s = requests.session()
            res = s.get(WGETDIR, auth=(user, pd), verify=False)
            requests.packages.urllib3.disable_warnings()
            if data[i]['whl_path'] != "":
                url = f"{WGETDIR}/{data[i]['whl_path']}"
                if not url.endswith(".html") and not url.endswith("/"):
                    url += "/"
                re_name = data[i]['whl_name'].replace('.whl', '\\.whl')
                name = rf"{re_name}"
                res = s.get(url, auth=(user, pd), verify=False)
                html = etree.HTML(res.text, parser=etree.HTMLParser())
                links = html.xpath("//a[@title]")
                if links:
                    for link_ in links:
                        title = link_.get("title", "")
                        href = link_.get("href", "")
                        if re.findall(name, title):
                            download_url = url+'/'+href
                            dowmloaded = requests.get(download_url, stream=True, auth=(user, pd), verify=False)
                            with open(title, 'wb') as fd:
                                shutil.copyfileobj(dowmloaded.raw, fd)
                            print(f"Download {title} success!")

            if 'tar_path' in data[i].keys():
                if data[i]['tar_path'] != '':
                    url = f"{WGETDIR}/{data[i]['tar_path']}"
                    if not url.endswith(".html") and not url.endswith("/"):
                        url += "/"
                    re_name = data[i]['tar_name'].replace('.tar.gz', '\\.tar\\.gz')
                    name = rf"{re_name}"
                    res = s.get(url, auth=(user, pd), verify=False)
                    html = etree.HTML(res.text, parser=etree.HTMLParser())
                    links = html.xpath("//a[@title]")
                    if links:
                        for link_ in links:
                            title = link_.get("title", "")
                            href = link_.get("href", "")
                            if re.findall(name, title):
                                download_url = url+'/'+href
                                dowmloaded = requests.get(download_url, stream=True, auth=(user, pd), verify=False)
                                with open(title, 'wb') as fd:
                                    shutil.copyfileobj(dowmloaded.raw, fd)
                                print(f"Download {title} success!")

        elif version != "daily":
            if data[i]['whl_path'] != "":
                release_url = f"https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}"
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                download_url = release_url + data[i]['whl_path'] + data[i]['whl_name']
                dowmloaded = requests.get(download_url, stream=True, verify=False)
                with open(data[i]['whl_name'], 'wb') as fd:
                    shutil.copyfileobj(dowmloaded.raw, fd)
                print(f"Download {data[i]['whl_name']} success!")
            if 'tar_path' in data[i].keys():
                if data[i]['tar_path'] != '':
                    release_url = f"https://ms-release.obs.cn-north-4.myhuaweicloud.com/{version}"
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    download_url = release_url + data[i]['tar_path'] + data[i]['tar_name']
                    dowmloaded = requests.get(download_url, stream=True, verify=False)
                    with open(data[i]['tar_name'], 'wb') as fd:
                        shutil.copyfileobj(dowmloaded.raw, fd)
                    print(f"Download {data[i]['tar_name']} success!")

    # 安装opencv-python额外依赖
    cmd = ["pip", "install", "opencv-python"]
    process = subprocess.Popen(cmd, stderr=subprocess.PIPE, encoding="utf-8")
    process.communicate()
    process.wait()

    # 安装各个组件的需要的安装包
    os.chdir(WHLDIR)

    whls = os.listdir()
    if whls:
        for i in whls:
            if "mindpandas" in i and "cp38-cp38" in i:
                os.rename(os.path.join(WHLDIR, i), os.path.join(WHLDIR, i.replace('cp38-cp38', 'cp37-cp37m')))
                cmd_install = ["pip", "install", i.replace('cp38-cp38', 'cp37-cp37m')]
                subprocess.run(cmd_install)
            elif "tar.gz" not in i:
                cmd_install = ["pip", "install", i]
                subprocess.run(cmd_install)
            else:
                os.environ["LITE_PACKAGE_PATH"] = os.path.join(WHLDIR, i)

    ERRORLOGDIR = f"{WORKDIR}/errorlog/"


    flush(OUTPUTDIR)
    flush(ERRORLOGDIR)
    error_lists = []
    failed_list = []
    failed_name_list = []

    # 遍历ArraySource开始生成html
    # pylint: disable=R1702
    for i in ArraySource:
        if "tutorials" in i:
            os.chdir(os.path.join(DOCDIR, "../../", i))
        else:
            os.chdir(os.path.join(DOCDIR, "../../docs", i))
        subprocess.run(["pip", "install", "-r", "requirements.txt"])
        if os.path.exists("source_zh_cn"):
            # 输出中文
            try:
                print(f"当前输出-{i}- 的-中文-版本---->")
                with open("Makefile", "r+") as f:
                    content = f.read()
                    content_mod = content.replace("source_en", "source_zh_cn")\
                        .replace("build_en", "build_zh_cn")
                    f.seek(0)
                    f.truncate()
                    f.write(content_mod)
                subprocess.run(["make", "clean"])
                cmd_make = ["make", "html"]
                process = subprocess.Popen(cmd_make, stderr=subprocess.PIPE, encoding="utf-8")
                _, stderr = process.communicate()
                process.wait()
                if stderr:
                    for j in stderr.split("\n"):
                        if ": WARNING:" in j:
                            error_lists.append(deal_err(j))
                if process.returncode != 0:
                    print(f"{i} 的 中文版本运行失败")
                    print(stderr)
                    failed_list.append(stderr)
                    failed_name_list.append(f'{i}的中文版本')
                else:
                    if i == "mindspore":
                        TARGET = f"{OUTPUTDIR}/docs/zh-CN/{ArraySource[i]}"
                        os.makedirs(os.path.dirname(TARGET), exist_ok=True)
                        shutil.copytree("build_zh_cn/html", TARGET)
                    else:
                        TARGET = f"{OUTPUTDIR}/{i}/zh-CN/{ArraySource[i]}"
                        os.makedirs(os.path.dirname(TARGET), exist_ok=True)
                        shutil.copytree("build_zh_cn/html", TARGET)
            # pylint: disable=W0702
            except:
                print(f"{i} 的 中文版本运行失败")

        # 输出英文
        if os.path.exists("source_en"):
            try:
                print(f"当前输出-{i}- 的-英文-版本---->")
                with open("Makefile", "r+") as f:
                    content = f.read()
                    content_mod = content.replace("source_zh_cn", "source_en")\
                        .replace("build_zh_cn", "build_en")
                    f.seek(0)
                    f.truncate()
                    f.write(content_mod)

                subprocess.run(["make", "clean"])
                cmd_make = ["make", "html"]
                process = subprocess.Popen(cmd_make, stderr=subprocess.PIPE, encoding="utf-8")
                _, stderr = process.communicate()
                process.wait()
                if stderr:
                    for j in stderr.split("\n"):
                        if ": WARNING:" in j:
                            error_lists.append(deal_err(j))
                if process.returncode != 0:
                    print(f"{i} 的 英文 版本运行失败")
                    with open("err_cn.log", "w") as f:
                        f.write(stderr)
                    failed_name_list.append(f'{i}的英文版本')
                else:
                    if i == "mindspore":
                        TARGET = f"{OUTPUTDIR}/docs/en/{ArraySource[i]}"
                        os.makedirs(os.path.dirname(TARGET), exist_ok=True)
                        shutil.copytree("build_en/html", TARGET)
                    else:
                        TARGET = f"{OUTPUTDIR}/{i}/en/{ArraySource[i]}"
                        os.makedirs(os.path.dirname(TARGET), exist_ok=True)
                        shutil.copytree("build_en/html", TARGET)
            # pylint: disable=W0702
            except:
                print(f"{i} 的 英文版本运行失败")

    # 将每个组件的warning写入文件
    if error_lists:
        with open(os.path.join(WORKDIR, 'err.txt'), 'wb') as f:
            pickle.dump(error_lists, f)

    # 将构建失败组件的报错信息写入文件
    if failed_list:
        with open(os.path.join(ERRORLOGDIR, 'fail.txt'), 'wb') as f:
            pickle.dump(failed_list, f)

    # 打印失败的组件
    print("构建完成！异常如下：")
    if failed_name_list:
        for j in failed_name_list:
            print(j, "失败")

    # 计时结束
    time_stop = time.perf_counter()

    all_time = time_stop - time_start
    minutes, seconds = divmod(all_time, 60)
    print(f"运行完成，总计耗时 {minutes}分 {seconds}秒.")

if __name__ == "__main__":
    # 配置一个工作目录
    try:
        MAINDIR = os.environ["work_dir"]
    except KeyError:
        MAINDIR = os.path.dirname(os.path.abspath(__file__))

    DOCDIR = os.path.dirname(os.path.abspath(__file__))

    # 添加命令行参数以供使用
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, default="daily") # release as 1.9.0 or 1.8.1 or 1.8.0
    parser.add_argument('--user', type=str, default="") # repo url username
    parser.add_argument('--pd', type=str, default="") # repo url password
    parser.add_argument('--wgetdir', type=str, default="") # repo url
    args = parser.parse_args()

    password = args.pd

    # 替换linux下命令行不允许的类似!#前面的反斜杠
    password = password.replace('\\', '')

    # git 克隆仓保存路径
    REPODIR = f"{MAINDIR}/repository"

    # 开始执行
    try:
        main(version=args.version, user=args.user, pd=password, WGETDIR=args.wgetdir)
    except (KeyboardInterrupt, SystemExit):
        print("程序即将终止....")
        time.sleep(1)
