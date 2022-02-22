"""
链接检测工具
"""
import sys
import os
import re
import json
import threading
import logging
import requests

# pylint: disable=W0150,W0703

logging.basicConfig(level=logging.WARNING)
requests.packages.urllib3.disable_warnings()

lock = threading.Lock()

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

def get_urls(content):
    """
    获取字符串中的链接
    """
    re_url = r"(https:\/\/|http:\/\/|ftp:\/\/)([\w\-\.,@?^=%&amp;\n:\!/~\+#]*[\w\-\@?^=%&amp;/~\+#])?"
    url_list = []
    urls = re.findall(re_url, content)
    for url in urls:
        url_list.append(url[0]+url[1].split("\n\n")[0].replace("\n", ""))
    return url_list

def check_url_status(url):
    """
    检查链接的状态码
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0"}
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

def run_check(file):
    """
    检测文件中的urls链接
    """
    data = get_content(file)
    file_urls = get_urls(data)
    white_urls = get_white_urls()
    urls = set(file_urls) - set(white_urls)
    pool = []
    for url in urls:
        k = threading.Thread(target=update_url_status_to_json, args=(url,))
        k.start()
        pool.append(k)
    for j in pool:
        j.join()

    generate_info(file)
    if os.path.exists("url_status.json"):
        os.remove("url_status.json")

def get_white_urls(white_file="filter_linklint.txt"):
    """获取白名单中的链接"""
    for i in sys.argv[1:]:
        if "--white_path=" in i:
            white_file = i.split("=")[-1]
    if os.path.exists(white_file):
        try:
            with open(white_file, "r", encoding="utf-8") as f:
                urls = f.readlines()
        except Exception:
            with open(white_file, "r", encoding="GBK") as f:
                urls = f.readlines()
        urls = [u.replace("\n", "") for u in urls]
    else:
        urls = []
    return urls

def generate_info(file):
    """
    输出404链接的信息
    """
    if os.path.exists("url_status.json"):
        with open("url_status.json", "r") as f:
            url_status = json.load(f)
    else:
        url_status = {"https://www.mindspore.cn": 200}
    try:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        with open("filter_linklint.txt", "r", encoding="GBK") as f:
            lines = f.readlines()
    for line_num, line_content in enumerate(lines, 1):
        for i in get_urls(line_content):
            if url_status[i] == 404:
                msg = "{}:line_{}:{}: Error link in the line! {}".format(file, line_num, url_status[i], i)
                if "gitee.com" in i:
                    logging.warning(msg)
                else:
                    logging.error(msg)

if __name__ == "__main__":
    for check_path_ in sys.argv[1:]:
        extension = ["md", "py", "rst", "ipynb", "js", "html", "c", "cc", "txt"]
        if os.path.isfile(check_path_) and check_path_.split(".")[-1] in extension:
            run_check(check_path_)
        elif os.path.isdir(check_path_):
            check_f_ = [file for file in find_file(check_path_, files=[]) if file.split(".")[-1] in extension]
            for one_f in check_f_:
                run_check(one_f)
