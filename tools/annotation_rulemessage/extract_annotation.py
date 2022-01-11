"""Extract file comments or walk through the file in the folder then extract file comments"""
import re
import os
import sys

def run(file_path):
    """export result"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()

    fileinput_path = str(file_path).split(sys.argv[1])[1]
    all_data = re.findall(r'"""([\s\S]*?)"""', data)
    file_name = file_path.split("/")[-1].split(".")[0]
    out_path = "annotation"+"/".join(fileinput_path.split("/")[:-1])+"/{}-annotation.txt".format(file_name)
    out_dir = "/".join(out_path.split("/")[:-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(all_data)
