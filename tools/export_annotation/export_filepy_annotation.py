"""Extract file comments or walk through the file in the folder then extract file comments"""
import re
import os
import sys

def find_files(path, files=None):
    file_name = os.listdir(path)
    is_file = [path+"/"+i for i in file_name if os.path.isfile(path+"/"+i)]
    is_dir = [j for j in file_name if os.path.isdir(path+"/"+j)]
    if is_file:
        files.extend(is_file)
    if is_dir:
        for k in is_dir:
            find_files(path + "/" + k, files)
    return files

def run(file_path):
    """export result"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    all_data = re.findall(r'"""([\s\S]*?)"""', data)
    file_name = file_path.split("/")[-1].split(".")[0]
    out_path = "annotation/"+"/".join(str(file_path).split("/")[:-1])+"/{}.txt".format(file_name)
    out_dir = "/".join(out_path.split("/")[:-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(all_data)

if __name__ == "__main__":
    if not os.path.exists('annotation'):
        os.mkdir('annotation')
    for check_f in sys.argv[1:]:
        if os.path.isfile(check_f):
            run(check_f)
        elif os.path.isdir(check_f):
            py_f = [file for file in find_files(check_f, files=[]) if file.endswith(".py")]
            for one_py_f in py_f:
                run(one_py_f)
    