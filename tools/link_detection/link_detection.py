import subprocess
import re
import requests
import urllib3
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

def get_all_file(check_path):
    '''
    get all the files in the directory.
    '''
    cmd = 'find %s -type f' %check_path
    res = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    all_file_list = res.stdout.read().decode('utf-8').split('\n')
    del all_file_list[-1]
    return all_file_list

def get_all_link(all_file_list):
    '''
    get all the links in all the files.
    '''
    re_rule = "(https:\/\/)([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?"
    for i in all_file_list:
        i = i.split('/', 1)[1].replace('/', ':/', 1)
        try:
            with open(i, 'r', encoding='utf-8') as f:
                data = f.read()
                link_list = []
                urls = re.findall(re_rule, data, re.S)
                if urls:
                    for url in urls:
                        link_list.append(url[0]+url[1])
                if link_list:
                    dic[i] = link_list
        except Exception:
            continue

def get_status(addr):
    '''
    Request the link and write different results to different files.
    '''
    try:
        link_path, link_addr, file_404, file_exception, mutexA, mutexB = addr[0], addr[1], addr[2], addr[3], addr[4], addr[5]
        response = requests.get(link_addr, headers=headers, verify=False, timeout=5)
        print(link_addr)
        print(response.status_code)
        if response.status_code != 200:
            mutexA.acquire()
            file_404.write('链接所在路径: %s' %link_path)
            file_404.write('\n')
            file_404.write('链接地址：%s' %link_addr)
            file_404.write('\n')
            file_404.write('链接的状态码：%s' %response.status_code)
            file_404.write('\n\n\n\n\n')
            mutexA.release()
    except Exception :
        print('exception!')
        mutexB.acquire()
        file_exception.write('链接所在路径: %s' %link_path)
        file_exception.write('\n')
        file_exception.write('链接地址：%s' %link_addr)
        file_exception.write('\n\n\n\n\n')
        mutexB.release()

def multi_threading():
    '''
    open multithreading to finish tasks concurrently, do not send a request to the download link, write it directly.
    '''
    for i in dic:
        link_list = list(set(dic[i]))
        for j in link_list:
            if j.endswith('.whl') or j.endswith('.gz'):
                f3.write('链接所在路径: %s' %i)
                f3.write('\n')
                f3.write('链接地址：%s' %j)
                f3.write('\n\n\n\n\n')
                continue
            pool.submit(get_status, (i, j, f1, f2, mutexA, mutexB))
    pool.shutdown()
    f1.close()
    f2.close()
    f3.close()

def main():
    all_file_list = get_all_file(check_path)
    get_all_link(all_file_list)
    multi_threading()

if __name__ == '__main__':
    check_path = input('请输入您要检测的绝对路径：').strip()
    dic = {}
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36'}
    pool = ThreadPoolExecutor(500)
    f1 = open('./404.txt', 'w', encoding='utf-8')
    f2 = open('./exception.txt', 'w', encoding='utf-8')
    f3 = open('./slow.txt', 'w', encoding='utf-8')
    mutexA = Lock()
    mutexB = Lock()
    main()
