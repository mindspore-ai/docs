'''pyfile extract annotation annotation_txt, commandline annotation_txt rulemessage_txt, rulemessage_txt to html'''
import os
import sys
import extract_annotation as ext
import commandline_extract_rulemessage as cer
import txttohtml as tth

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

if __name__ == "__main__":
    for check_f in sys.argv[1:]:
        print(sys.argv[1:])
        if os.path.isfile(check_f) & check_f.endswith(".py"):
            ext.run(check_f)
        elif os.path.isdir(check_f):
            all_f = [file for file in find_files(check_f, []) if file.endswith(".py")]
            for one_f in all_f:
                ext.run(one_f)
    for check_f in sys.argv[1:]:
        print(sys.argv[1:])
        if os.path.isfile(check_f) & check_f.endswith("-annotation.txt"):
            cer.run(check_f)
        elif os.path.isdir(check_f):
            all_f = [file for file in find_files(check_f, []) if file.endswith("-annotation.txt")]
            for one_f in all_f:
                cer.run(one_f)
    for check_f in sys.argv[1:]:
        print(sys.argv[1:])
        if os.path.isfile(check_f) & check_f.endswith("-rulemessage.txt"):
            tth.run(check_f)
        elif os.path.isdir(check_f):
            all_f = [file for file in find_files(check_f, []) if file.endswith("-rulemessage.txt")]
            for one_f in all_f:
                tth.run(one_f)
    for check_f in sys.argv[1:]:
        result = '<p>规则参考链接：https://community.languagetool.org/rule/list?offset=0</p>\n'
        with open("./allrule.html", "a", encoding="utf-8") as p:
            p.writelines(result)
        if os.path.isfile(check_f) & check_f.endswith(".html"):
            with open(check_f, "r", encoding="utf-8") as f:
                data = f.read()
            with open("./allrule.html", "a", encoding="utf-8") as p:
                p.writelines(data)
        elif os.path.isdir(check_f):
            all_f = [file for file in find_files(check_f, []) if file.endswith(".html")]
            for one_f in all_f:
                with open(one_f, "r", encoding="utf-8") as f:
                    data = f.read()
                with open("./allrule.html", "a", encoding="utf-8") as p:
                    p.writelines(data)
                