"""
链接检测工具
"""
import os
import re
import json
import threading
import argparse
import requests

# pylint: disable=W0150,W0703

parser = argparse.ArgumentParser(description="link ci")
parser.add_argument("-c", "--check-list", type=str, default="./check_list.txt",
                    help="List of files that need to be checked for Link validity")
parser.add_argument("-w", "--white-list", type=str, default="./filter_linklint.txt", help="Whitelisted link list")
parser.add_argument("-p", "--path", type=str, default=None, help="check path")

white_example = [
    "https://gluebenchmark.com/tasks",
    "https://developer.huawei.com/repo/"
]

args = parser.parse_args()

requests.packages.urllib3.disable_warnings()
lock = threading.Lock()


def get_all_files():
    """
    获取所有需要检测的文件
    """
    extension = ["md", "py", "ipynb", "c", "cc", "js", "rst"]
    check_list_info1 = get_check_info(info_type="check_list")
    check_list_info2 = args.path.split(",") if args.path else []
    check_list_info = check_list_info1 + check_list_info2
    file_list = []
    for i in check_list_info:
        if os.path.isfile(i):
            file_list.append(i)
        elif os.path.isdir(i):
            file_list1 = [j for j in find_file(i, []) if "/." not in j and j.split(".")[-1] in extension]
            file_list.extend(file_list1)
        else:
            print(f"The {i} is not exist")
    return file_list

def get_all_urls(file_list):
    """获取所有文件中的链接并去重"""
    urls_list = []
    for i in file_list:
        urls_list += get_file_urls(i)
    return list(set(urls_list))

def find_file(path, files=None):
    """递归遍历path中的所有文件"""
    file_name = os.listdir(path)
    is_file = [path + "/" + i for i in file_name if os.path.isfile(path + "/" + i)]
    is_dir = [j for j in file_name if os.path.isdir(path + "/" + j)]
    if is_file:
        files.extend(is_file)
    if is_dir:
        for k in is_dir:
            find_file(path+"/"+k, files)
    return files

def get_file_urls(file, isfile=True):
    """
    获取字符串中的链接
    """
    re_url = r"(https:\/\/|http:\/\/|ftp:\/\/)([\w\-\.@?^=%&amp;:\!/~\+#]*[\w\-\@?^=%&amp;/~\+#])?"
    url_list = []
    if isfile:
        content = get_content(file)
        if file.endswith(".py") or file.endswith(".rst"):
            lines = content.replace("\\", "").split("\n")
            lines = [i.strip() for i in lines]
            content = "\n".join(lines).replace("\n\n", " ").replace("\n", "")
    else:
        content = file
    urls = re.findall(re_url, content)
    for url in urls:
        url_list.append(url[0]+url[1])
    return url_list

def check_url_status(url):
    """
    检查链接的状态码
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"}
    if url.startswith("https://pki.consumer.huawei.com/ca/"):
        return 200
    try:
        res = requests.get(url, stream=True, headers=headers, timeout=5, verify=False)
        status = res.status_code
    except Exception:
        status = "failed connect"
    finally:
        return status

def get_content(file_path):
    """
    获取文档的内容
    """
    contents = ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            contents = f.read()
    except Exception:
        with open(file_path, "r", encoding="GBK") as f:
            contents = f.read()
    finally:
        return contents

def update_json(data, json_file):
    """
    更新字典数据到本地的json文件中
    """
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            json_content = json.load(f)
        json_content.update(data)
    else:
        json_content = data
    with open(json_file, "w") as f:
        json.dump(json_content, f, indent=4)

def update_url_status_to_json(url):
    """
    检测链接的状态码并将链接与状态码的键值对存入到url_status.json中
    """
    status = check_url_status(url)
    data = {url: status}
    lock.acquire()
    update_json(data, "url_status.json")
    lock.release()

def is_white_url(re_url, url):
    """
    判断是否符合模糊匹配的链接
    """
    results = re.findall(re_url, url)
    if results:
        return results[0] == url
    return False

def run_check(all_files):
    """
    检测文件中的urls链接
    """
    all_urls = get_all_urls(all_files)
    white_urls = get_check_info(info_type="white_list") + white_example
    urls = set(all_urls) - set(white_urls)
    re_white_urls = [re_url.replace(".", r"\.").replace("*", ".*") for re_url in white_urls if "*" in re_url]
    white_url_save = {}
    for i in re_white_urls:
        white_url_save.update({j: 200 for j in urls if is_white_url(i, j)})
        urls -= white_url_save.keys()
    pool = []
    for url in urls:
        k = threading.Thread(target=update_url_status_to_json, args=(url,))
        k.start()
        pool.append(k)
    for j in pool:
        j.join()
    update_json(white_url_save, "url_status.json")

def location_error_line(file, url):
    """
    定位问题链接在文件中的位置
    """
    msg = []
    try:
        with open(file, "r", encoding="utf-8") as f:
            infos = f.readlines()
    except Exception:
        with open(file, "r", encoding="GBK") as f:
            infos = f.readlines()

    if file.endswith(".py") or file.endswith(".rst"):
        contents = get_content(file)
        if url in contents:
            for line_num, line in enumerate(infos, 1):
                line_urls = get_file_urls(line, isfile=False)
                if url in line_urls:
                    msg.append("{}: line_{}: Error link: {}".format(file, line_num, url))
        else:
            left = contents.replace("\n", "").replace("\\", "").replace(" ", "").split(url)[0]
            for line_num, line in enumerate(infos, 1):
                if line.replace("\n", "").replace("\\", "").replace(" ", "") not in left:
                    msg.append("{}: line_{}: Link format error: {}".format(file, line_num, url))
                    break
    else:
        for line_num, line in enumerate(infos, 1):
            line_urls = get_file_urls(line, isfile=False)
            if url in line_urls:
                msg.append("{}: line_{}: Error link: {}".format(file, line_num, url))
    return msg

def generator_report(all_files):
    """生成报告"""
    msg_list = []
    white_urls = get_check_info(info_type="white_list") + white_example
    with open("url_status.json", "r") as f:
        url_status = json.load(f)
    for file_name in all_files:
        urls = get_file_urls(file_name)
        urls = list(set(urls))
        for u in urls:
            if u not in white_urls:
                if url_status[u] == 404:
                    msg_list.extend(location_error_line(file_name, u))

    for msg in msg_list:
        if "gitee.com" in msg:
            print(f"WARRING:{msg}")
        else:
            print(f"ERROR:{msg}")

def get_check_info(info_type="check_list"):
    """获取需要检测的信息"""
    if info_type == "white_list":
        info_file = args.white_list
    else:
        info_file = args.check_list
    if os.path.exists(info_file):
        try:
            with open(info_file, "r", encoding="utf-8") as f:
                infos = f.readlines()
        except Exception:
            with open(info_file, "r", encoding="GBK") as f:
                infos = f.readlines()
        infos_list = [info.replace("\n", "") for info in infos]
    else:
        infos_list = []
    return infos_list

if __name__ == "__main__":
    all_file = get_all_files()
    run_check(all_file)
    if os.path.exists("url_status.json"):
        generator_report(all_file)
        os.remove("url_status.json")
