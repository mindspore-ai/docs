"""
使用json文件自动化生成mindspore各组件的html页面
"""
import argparse
import copy
import datetime
import glob
import json
import os
import pickle
import re
import shutil
import subprocess
import time
from multiprocessing import Pool
import requests
import sphinx
import urllib3
from git import Repo
from lxml import etree
from replace_html_menu import replace_html_menu, modify_menu_num

# 下载仓库
def git_clone(repo_url, repo_dir):
    if not os.path.exists(repo_dir):
        print("Cloning repo.....")
        os.makedirs(repo_dir, exist_ok=True)
        Repo.clone_from(repo_url, repo_dir, branch="master")
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

def deal_err(err, pylib_dir):
    extra_str_re = re.compile(r"\[3.*?m")
    workdir_re = re.compile(rf"{REPODIR}")
    pythonlib_re = re.compile(rf"{pylib_dir}")
    err_new = extra_str_re.sub('', err)
    err_new = workdir_re.sub('', err_new)
    err_new = pythonlib_re.sub('', err_new)
    return err_new

def flush(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def generate_version_json(repo_name, branch, js_data, version_flag, target_path):
    """
    基于base_version.json文件给每个组件生成对应的version.json文件。
    """
    for d in range(len(js_data)):
        if js_data[d]['repo_name'] == repo_name:
            write_content = copy.deepcopy(js_data[d])
            if not write_content['version']:
                write_content['version'] = branch
            write_content.pop("repo_name", None)
            if js_data[d]['repo_name'] != 'mindspore':
                filename = js_data[d]['repo_name']
            else:
                filename = "docs"
            if not version_flag and "submenu" in write_content.keys():
                for url in write_content["submenu"]["zh"]:
                    url["url"] = url["url"].replace('/master/', f'/{branch}/')
                for url in write_content["submenu"]["en"]:
                    url["url"] = url["url"].replace('/master/', f'/{branch}/')
            with open(os.path.join(target_path, f"{filename}_version.json"), 'w+', encoding='utf-8') as g:
                json.dump(write_content, g, indent=4)
            break

#######################################
# 运行检测
#######################################
def main(version, user, pd, WGETDIR, release_url, generate_list):

    print(f"开始构建{version}版本html....")

    # 保存需要生成html的文件夹及对应版本号
    ArraySource = dict()

    # 不同版本保存路径
    WORKDIR = f"{MAINDIR}/{version}"

    # html文档页面保存路径
    OUTPUTDIR = f"{WORKDIR}/output"

    # 各个组件安装包下载保存路径
    WHLDIR = f"{WORKDIR}/whlpkgs"

    # python安装包文件夹位置
    pythonlib_dir = os.path.dirname(os.path.dirname(sphinx.__file__))

    # 删除sphinx中多余的语言文件
    mo_path = os.path.join(pythonlib_dir, 'locale/zh_CN/LC_MESSAGES/sphinx.mo')
    if os.path.exists(mo_path):
        os.remove(mo_path)

    # 开始计时
    time_start = time.perf_counter()

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome\
               /115.0.0.0 Safari/537.36"}

    # 读取json文件数据
    if version == "daily":
        flag_dev = 1
        with open(os.path.join(os.path.dirname(__file__), "daily.json"), 'r+', encoding='utf-8') as g:
            data = json.load(g)
    else:
        flag_dev = 0
        with open(os.path.join(os.path.dirname(__file__), "version.json"), 'r+', encoding='utf-8') as g:
            data = json.load(g)

    with open(os.path.join(os.path.dirname(__file__), "base_version.json"), 'r+', encoding='utf-8') as g:
        data_b = json.load(g)

    target_version = f"{MAINDIR}/{version}_version"
    flush(target_version)

    flush(WHLDIR)
    # 遍历json数据做好生成html前的准备
    # pylint: disable=R1702
    msc_branch = ''

    # 单独生成时提取msc仓的分支
    if generate_list:
        for i in range(len(data)):
            if data[i]['name'] == 'mindscience':
                msc_branch = data[i]["branch"]

    for i in range(len(data)):

        # 克隆仓库与配置环境变量
        repo_name = data[i]['name'].replace('_', '-')
        repo_url = f"https://gitee.com/mindspore/{repo_name}.git"
        repo_path = f"{REPODIR}/{data[i]['name']}"
        branch_ = data[i]["branch"]

        if data[i]['environ'] == "MS_PATH":
            repo_url = "https://gitee.com/mindspore/mindspore.git"
            repo_path = f"{REPODIR}/mindspore"
        elif data[i]['environ'] == "MSC_PATH":
            repo_url = "https://gitee.com/mindspore/mindscience.git"
            repo_path = f"{REPODIR}/mindscience"
        elif data[i]['name'] == "devtoolkit":
            repo_url = "https://gitee.com/mindspore/ide-plugin.git"
            repo_path = f"{REPODIR}/ide-plugin"
        elif data[i]['name'] == "reinforcement":
            repo_url = "https://github.com/mindspore-lab/mindrl.git"
            repo_path = f"{REPODIR}/mindrl"
        elif data[i]['name'] == "recommender":
            repo_url = "https://github.com/mindspore-lab/mindrec.git"
            repo_path = f"{REPODIR}/mindrec"

        # 判断是否需要单独生成某些组件
        if generate_list and data[i]['name'] not in generate_list:
            continue
        if data[i]['environ'] and branch_:
            os.environ[data[i]['environ']] = repo_path
            try:
                status_code = requests.get(repo_url, headers=headers).status_code
                if status_code == 200:
                    if not os.path.exists(repo_path):
                        git_clone(repo_url, repo_path)
                    if data[i]['environ'] == "MSC_PATH":
                        if data[i]['name'] == "mindscience":
                            git_update(repo_path, branch_)
                        elif msc_branch:
                            git_update(repo_path, msc_branch)
                    else:
                        git_update(repo_path, branch_)
                    print(f'{repo_name}仓库克隆更新成功')
            except KeyError:
                print(f'{repo_name}仓库克隆或更新失败')

        # 组件仓内有.sh需提前运行
        if 'golden_stick' in repo_path:
            os.chdir(repo_path)
            cmd_reppath = ["sh", "./docs/adapte_to_docs.sh", f"{branch_}"]
            subprocess.run(cmd_reppath)

        if data[i]['name'] != "mindscience":
            generate_version_json(data[i]['name'], data[i]["branch"], data_b, flag_dev, target_version)

        # 卸载原来已有的安装包, 以防冲突
        if data[i]['uninstall_name']:
            cmd_uninstall = ["pip", "uninstall", "-y", f"{data[i]['uninstall_name']}"]
            subprocess.run(cmd_uninstall)

        os.chdir(WHLDIR)

        # 从网站下载各个组件需要的whl包或tar包
        if version == "daily" or flag_dev:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            s = requests.session()
            if data[i]['name'] == "reinforcement" or data[i]['name'] == "recommender":
                wgetdir = WGETDIR + "mindspore-lab"
            else:
                wgetdir = WGETDIR + "mindspore"
            res = s.get(wgetdir, auth=(user, pd), verify=False)
            requests.packages.urllib3.disable_warnings()
            # 下载组件whl包
            if data[i]['whl_path'] != ""  and 'whl_search' in data[i]:
                today_str = datetime.date.today().strftime('%Y%m%d')
                month_str = datetime.date.today().strftime('%Y%m')
                search_url = f"{wgetdir}/{data[i]['whl_path']}/{month_str}/{today_str}/"
                res = s.get(search_url, auth=(user, pd), verify=False)
                html = etree.HTML(res.text, parser=etree.HTMLParser())
                links = html.xpath("//a[@title]")
                url = ''
                if links:
                    for link_ in links[::-1]:
                        href = link_.get("href", "")
                        if href.startswith('dev_'):
                            url = search_url+href+data[i]['whl_search']
                            break
                if not url:
                    continue

                re_name = data[i]['whl_name'].replace('.whl', '\\.whl')
                name = rf"{re_name}"
                res = s.get(url, auth=(user, pd), verify=False)
                html = etree.HTML(res.text, parser=etree.HTMLParser())
                links = html.xpath("//a[@title]")
                if links:
                    for link_ in links:
                        title = link_.get("title", "")
                        href = link_.get("href", "")
                        if re.findall(name, title) and not os.path.exists(os.path.join(WHLDIR, title)):
                            download_url = url+href
                            downloaded = requests.get(download_url, stream=True, auth=(user, pd), verify=False)
                            with open(title, 'wb') as fd:
                                #shutil.copyfileobj(dowmloaded.raw, fd)
                                for chunk in downloaded.iter_content(chunk_size=512):
                                    if chunk:
                                        fd.write(chunk)
                            print(f"Download {title} success!")
                            time.sleep(1)

            elif data[i]['whl_path'] != "":
                url = f"{wgetdir}/{data[i]['whl_path']}"
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
                        if re.findall(name, title) and not os.path.exists(os.path.join(WHLDIR, title)):
                            download_url = url+href
                            downloaded = requests.get(download_url, stream=True, auth=(user, pd), verify=False)
                            with open(title, 'wb') as fd:
                                #shutil.copyfileobj(dowmloaded.raw, fd)
                                for chunk in downloaded.iter_content(chunk_size=512):
                                    if chunk:
                                        fd.write(chunk)
                            print(f"Download {title} success!")
                            time.sleep(1)

            # 下载其他需求的组件whl包
            if 'extra_whl_path' in data[i] and data[i]['extra_whl_path'] != "":
                url = f"{wgetdir}/{data[i]['extra_whl_path']}"
                if not url.endswith(".html") and not url.endswith("/"):
                    url += "/"
                re_name = data[i]['extra_whl_name'].replace('.whl', '\\.whl')
                name = rf"{re_name}"
                res = s.get(url, auth=(user, pd), verify=False)
                html = etree.HTML(res.text, parser=etree.HTMLParser())
                links = html.xpath("//a[@title]")
                if links:
                    for link_ in links:
                        title = link_.get("title", "")
                        href = link_.get("href", "")
                        if re.findall(name, title) and not os.path.exists(os.path.join(WHLDIR, title)):
                            download_url = url+href
                            downloaded = requests.get(download_url, stream=True, auth=(user, pd), verify=False)
                            with open(title, 'wb') as fd:
                                #shutil.copyfileobj(dowmloaded.raw, fd)
                                for chunk in downloaded.iter_content(chunk_size=512):
                                    if chunk:
                                        fd.write(chunk)
                            print(f"Download {title} success!")
                            time.sleep(1)

            if 'tar_path' in data[i].keys():
                if data[i]['tar_path'] != '':
                    url = f"{wgetdir}/{data[i]['tar_path']}"
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
                                download_url = url+href
                                downloaded = requests.get(download_url, stream=True, auth=(user, pd), verify=False)
                                with open(title, 'wb') as fd:
                                    #shutil.copyfileobj(dowmloaded.raw, fd)
                                    for chunk in downloaded.iter_content(chunk_size=512):
                                        if chunk:
                                            fd.write(chunk)
                                print(f"Download {title} success!")

        elif version != "daily":
            if data[i]['whl_path'] != "":
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                download_url = release_url + data[i]['whl_path'] + data[i]['whl_name']
                downloaded = requests.get(download_url, stream=True, verify=False)
                with open(data[i]['whl_name'], 'wb') as fd:
                    #shutil.copyfileobj(dowmloaded.raw, fd)
                    for chunk in downloaded.iter_content(chunk_size=512):
                        if chunk:
                            fd.write(chunk)
                print(f"Download {data[i]['whl_name']} success!")
            if 'extra_whl_path' in data[i] and data[i]['extra_whl_path'] != "":
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                download_url = release_url + data[i]['extra_whl_path'] + data[i]['extra_whl_name']
                downloaded = requests.get(download_url, stream=True, verify=False)
                with open(data[i]['extra_whl_name'], 'wb') as fd:
                    #shutil.copyfileobj(dowmloaded.raw, fd)
                    for chunk in downloaded.iter_content(chunk_size=512):
                        if chunk:
                            fd.write(chunk)
                print(f"Download {data[i]['extra_whl_name']} success!")
            if 'tar_path' in data[i].keys():
                if data[i]['tar_path'] != '':
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                    download_url = release_url + data[i]['tar_path'] + data[i]['tar_name']
                    downloaded = requests.get(download_url, stream=True, verify=False)
                    with open(data[i]['tar_name'], 'wb') as fd:
                        #shutil.copyfileobj(dowmloaded.raw, fd)
                        for chunk in downloaded.iter_content(chunk_size=512):
                            if chunk:
                                fd.write(chunk)
                    print(f"Download {data[i]['tar_name']} success!")

        # 特殊与一般性的往ArraySource中加入键值对
        if not branch_:
            continue
        html_branch = branch_
        if "html_version" in data[i]:
            html_branch = data[i]["html_version"]
        if data[i]['name'] == "lite":
            ArraySource[data[i]['name'] + '/docs'] = html_branch
            ArraySource[data[i]['name'] + '/api'] = html_branch
        elif data[i]['name'] == "tutorials":
            ArraySource[data[i]['name']] = html_branch
        elif data[i]['name'] == "mindspore":
            ArraySource[data[i]['name']] = html_branch
        elif data[i]['name'] == "mindscience":
            pass
        else:
            ArraySource[data[i]['name'] + '/docs'] = html_branch

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

    # 安装mistune额外依赖
    cmd = ["pip", "install", "mistune==2.0.4"]
    process_1 = subprocess.Popen(cmd, stderr=subprocess.PIPE, encoding="utf-8")
    process_1.communicate()
    process_1.wait()

    ERRORLOGDIR = f"{WORKDIR}/errorlog/"


    flush(OUTPUTDIR)
    flush(ERRORLOGDIR)
    error_lists = []
    failed_list = []
    failed_name_list = []

    replace_flag = 1
    # 遍历ArraySource开始生成html
    # pylint: disable=R1702
    for i in ArraySource:
        if "tutorials" in i:
            os.chdir(os.path.join(DOCDIR, "../../", i))
        else:
            os.chdir(os.path.join(DOCDIR, "../../docs", i))
        subprocess.run(["pip", "install", "-r", "requirements.txt"])

        try:
            if replace_flag:
                from docutils import nodes
                nodes_target = os.path.join(os.path.dirname(nodes.__file__), 'nodes.py')
                nodes_src = os.path.join(DOCDIR, '../../resource/sphinx_ext/nodes.txt')
                if os.path.exists(nodes_target):
                    os.remove(nodes_target)
                shutil.copy(nodes_src, nodes_target)

                html_base_target = os.path.join(os.path.dirname(nodes.__file__), 'writers/_html_base.py')
                with open(html_base_target, 'r+', encoding='utf-8') as h:
                    html_base_content = h.read()
                    html_base_content = html_base_content.replace(
                        'self.meta = [self.generator % docutils.__version__]', 'self.meta = []')
                    h.seek(0)
                    h.truncate()
                    h.write(html_base_content)

                registry_target = os.path.join(pythonlib_dir, 'sphinx', 'registry.py')
                with open(registry_target, 'r+', encoding='utf-8') as h:
                    registry_content = h.read()
                    registry_content = re.sub(r'([ ]+?)except VersionRequirementError as err:\n(?:.|\n|)+?from err',
                                              r'\1except VersionRequirementError as err:\n\1    metadata = {}',
                                              registry_content)
                    h.seek(0)
                    h.truncate()
                    h.write(registry_content)
                replace_flag = 0
        except ModuleNotFoundError:
            pass

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
                            error_lists.append(deal_err(j, pythonlib_dir))
                if process.returncode != 0:
                    print(f"{i} 的 英文 版本运行失败")
                    print(f"错误信息：\n{stderr}")
                    with open("err_cn.log", "w") as f:
                        f.write(stderr)
                    failed_list.append(stderr)
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

        # 输出中文
        if os.path.exists("source_zh_cn"):
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
                            error_lists.append(deal_err(j, pythonlib_dir))
                if process.returncode != 0:
                    print(f"{i} 的 中文版本运行失败")
                    print(f"错误信息：\n{stderr}")
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

def yield_files(directory):
    """
    逐个给予需要处理的文件路径
    """
    # pylint: disable=W0612
    for root, dirs, files_ in os.walk(directory):
        if f'{directory}/_' in root:
            continue
        for file in files_:
            if file.endswith('.html'):
                yield os.path.join(root, file)

def process_file(file_path):
    """
    替换文件内容（search页面路径）
    """
    try:
        with open(file_path, "r+", encoding='utf8') as g:
            content = g.read()

            new_docs = 'search.html'
            old_docs = '../search.html'
            new_content = content.replace(old_docs, new_docs)
            if new_content != content:
                g.seek(0)
                g.truncate()
                g.write(new_content)
    # pylint: disable=W0703
    except Exception:
        print(f"{file_path}替换失败")

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
    parser.add_argument('--release_url', type=str, default="") # repo url
    parser.add_argument('--theme', type=str, default="") # theme.css/js
    parser.add_argument('--single_generate', type=str, default="")
    args = parser.parse_args()

    password = args.pd

    # 替换linux下命令行不允许的类似!#前面的反斜杠
    password = password.replace('\\', '')

    if args.single_generate:
        generate_list_p = [x.strip() for x in args.single_generate.split(',')]
    else:
        generate_list_p = []

    # git 克隆仓保存路径
    REPODIR = f"{MAINDIR}/repository"

    # 开始执行
    try:
        # 主函数组件html构建
        main(version=args.version, user=args.user, pd=password, WGETDIR=args.wgetdir,
             release_url=args.release_url, generate_list=generate_list_p)

        # 替换页面左侧目录部分
        ms_path = f"{MAINDIR}/{args.version}/output/docs/zh-CN/master"
        if os.path.exists(ms_path):
            replace_html_menu(ms_path, os.path.join(DOCDIR, "../../docs/mindspore/source_zh_cn"))
            print('docs中文目录大纲调整完成！')
            replace_html_menu(ms_path.replace('zh-CN', 'en'), os.path.join(DOCDIR, "../../docs/mindspore/source_en"))
            print('docs英文目录大纲调整完成！')
            # 修改每个页面内搜索页面的链接
            pool = Pool(processes=4)
            files = yield_files(ms_path)
            pool.map(process_file, files)
            pool.close()
            pool.join()
            print('docs所有页面search链接已修改！')
        ts_path = f"{MAINDIR}/{args.version}/output/tutorials/zh-CN/master"
        if os.path.exists(ts_path):
            modify_menu_num(ts_path)
            print('tutorials中文目录大纲调整完成！')
            modify_menu_num(ts_path.replace('zh-CN', 'en'))
            print('tutorials英文目录大纲调整完成！')

        # 替换样式相关内容
        theme_list = []
        output_path = f"{MAINDIR}/{args.version}/output"
        version_path = f"{MAINDIR}/{args.version}_version/"
        for dir_name in os.listdir(output_path):
            if os.path.isfile(os.path.join(output_path, dir_name)):
                continue
            if dir_name == 'docs':
                theme_list.append(dir_name)
            elif dir_name == 'tutorials':
                # theme_list.append(dir_name + '/application')
                # theme_list.append(dir_name + '/experts')
                theme_list.append(dir_name)
            elif dir_name == 'lite':
                theme_list.append(dir_name + '/docs')
                theme_list.append(dir_name + '/api')
            else:
                theme_list.append(dir_name + '/docs')
        theme_path = args.theme
        for f_name in os.listdir(theme_path):
            if os.path.isfile(os.path.join(theme_path, f_name)):
                if os.path.exists(os.path.join(output_path, f_name)):
                    os.remove(os.path.join(output_path, f_name))
                shutil.copy(os.path.join(theme_path, f_name), os.path.join(output_path, f_name))
        old_searchtools_content = """docContent = htmlElement.find('[role=main]')[0];"""
        new_searchtools_content = """htmlElement.find('[role=main]').find('[itemprop=articleBody]').find('style').remove();
      docContent = htmlElement.find('[role=main]')[0];"""
        # pylint: disable=W0621
        for lg in ['en', 'zh-CN']:
            # pylint: disable=W0621
            for out_name in theme_list:
                try:
                    static_path_searchtools = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/searchtools.js")[0]
                    static_path_css = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/css/theme.css")[0]
                    static_path_js = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/js/theme.js")[0]
                    static_path_jquery = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/jquery.js")[0]
                    static_path_underscore = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/underscore.js")[0]
                    static_path_jquery_ = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/jquery-3.5.1.js")[0]
                    static_path_underscore_ = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/underscore-1.13.1.js")
                    static_path_underscore_ = static_path_underscore_[0]

                    static_path_css_badge = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/css/badge_only.css")[0]
                    static_path_js_badge = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/js/badge_only.js")[0]
                    static_path_js_html5p = \
                    glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/js/html5shiv-printshiv.min.js")[0]
                    static_path_js_html5 = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/js/html5shiv.min.js")[0]

                    static_path_version = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/js/")[0]
                    static_path_version = os.path.join(static_path_version, "version.json")
                    if 'lite' in out_name:
                        css_path = f"theme-{out_name.split('/')[0]}/theme.css"
                        js_path = f"theme-{out_name.split('/')[0]}/theme.js"
                    elif out_name == 'docs':
                        css_path = "theme-tutorials/theme.css"
                        js_path = "theme-tutorials/theme.js"
                    else:
                        css_path = "theme-docs/theme.css"
                        js_path = "theme-docs/theme.js"
                    static_path_new_css = os.path.join(theme_path, css_path)
                    static_path_new_js = os.path.join(theme_path, js_path)
                    static_path_new_jquery = os.path.join(theme_path, "update_js", "jquery.js")
                    static_path_new_underscore = os.path.join(theme_path, "update_js", "underscore.js")
                    out_name_1 = out_name.split('/')[0]
                    static_path_new_version = os.path.join(version_path, f"{out_name_1}_version.json")
                    fonts_dir_1 = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/fonts/")
                    fonts_dir_2 = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/css/fonts/")
                    if fonts_dir_1 and os.path.exists(fonts_dir_1[0]):
                        shutil.rmtree(fonts_dir_1[0])
                    if fonts_dir_2 and os.path.exists(fonts_dir_2[0]):
                        shutil.rmtree(fonts_dir_2[0])
                    if os.path.exists(static_path_css):
                        os.remove(static_path_css)
                    shutil.copy(static_path_new_css, static_path_css)
                    if os.path.exists(static_path_js):
                        os.remove(static_path_js)
                    shutil.copy(static_path_new_js, static_path_js)
                    if os.path.exists(static_path_version):
                        os.remove(static_path_version)
                    shutil.copy(static_path_new_version, static_path_version)
                    if os.path.exists(static_path_jquery):
                        os.remove(static_path_jquery)
                    shutil.copy(static_path_new_jquery, static_path_jquery)
                    if os.path.exists(static_path_underscore):
                        os.remove(static_path_underscore)
                    shutil.copy(static_path_new_underscore, static_path_underscore)
                    if os.path.exists(static_path_jquery):
                        os.remove(static_path_jquery_)
                    if os.path.exists(static_path_underscore_):
                        os.remove(static_path_underscore_)
                    if os.path.exists(static_path_css_badge):
                        os.remove(static_path_css_badge)
                    if os.path.exists(static_path_js_badge):
                        os.remove(static_path_js_badge)
                    if os.path.exists(static_path_js_html5p):
                        os.remove(static_path_js_html5p)
                    if os.path.exists(static_path_js_html5):
                        os.remove(static_path_js_html5)
                    # 去除搜索页面冗余样式展示
                    if os.path.exists(static_path_searchtools):
                        with open(static_path_searchtools, 'r+', encoding='utf-8') as k:
                            searchtools_content = k.read()
                            if new_searchtools_content not in searchtools_content:
                                new_content_s = searchtools_content.replace(old_searchtools_content,
                                                                            new_searchtools_content)
                            if new_content_s != searchtools_content:
                                k.seek(0)
                                k.truncate()
                                k.write(new_content_s)

                # pylint: disable=W0702
                # pylint: disable=W0703
                except Exception as e:
                    print(f'替换{out_name}下{lg}样式文件失败!\n{e}')
                    continue
        print(f'替换样式文件成功!')
    except (KeyboardInterrupt, SystemExit):
        print("程序即将终止....")
        time.sleep(1)
