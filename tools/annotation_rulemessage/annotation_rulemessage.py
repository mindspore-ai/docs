'''python exe ext, cer, tth'''
import os
import sys
import extract_annotation as ext
import commandline_extract_rulemessage as cer
import txtoutput as tth

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
    if not os.path.exists('annotation'):
        os.mkdir('annotation')
    if not os.path.exists('rulemessage'):
        os.mkdir('rulemessage')
    if not os.path.exists('formatrule'):
        os.mkdir('formatrule')
    for check_f in sys.argv[1:]:
        if os.path.isfile(check_f) & check_f.endswith(".py"):
            ext.run(check_f)
        elif os.path.isdir(check_f):
            all_f = [file for file in find_files(check_f, []) if file.endswith(".py")]
            for one_f in all_f:
                ext.run(one_f)
    for check_f in os.listdir('./annotation'):
        check_f = 'annotation/' + check_f
        if os.path.isfile(check_f) & check_f.endswith("-annotation.txt"):
            cer.run(check_f)
        elif os.path.isdir(check_f):
            all_f = [file for file in find_files(check_f, []) if file.endswith("-annotation.txt")]
            for one_f in all_f:
                cer.run(one_f)
    for check_f in os.listdir('./rulemessage'):
        check_f = 'rulemessage/' + check_f
        if os.path.isfile(check_f) & check_f.endswith("-rulemessage.txt"):
            tth.run(check_f)
        elif os.path.isdir(check_f):
            all_f = [file for file in find_files(check_f, []) if file.endswith("-rulemessage.txt")]
            for one_f in all_f:
                tth.run(one_f)
    for check_f in os.listdir('./formatrule'):
        result = 'Rule reference link `https://community.languagetool.org/rule/list?offset=0` \n'
        with open("./tools/annotation_rulemessage/allrule.txt", "w", encoding="utf-8") as p:
            p.writelines(result)
        check_f = 'formatrule/' + check_f
        if os.path.isfile(check_f) & check_f.endswith("-outputrule.txt"):
            with open(check_f, "r", encoding="utf-8") as f:
                data = f.read()
            with open("./tools/annotation_rulemessage/allrule.txt", "a", encoding="utf-8") as p:
                p.writelines(data)
        elif os.path.isdir(check_f):
            all_f = [file for file in find_files(check_f, []) if file.endswith("-outputrule.txt")]
            for one_f in all_f:
                with open(one_f, "r", encoding="utf-8") as f:
                    data = f.read()
                with open("./tools/annotation_rulemessage/allrule.txt", "a", encoding="utf-8") as p:
                    p.writelines(data)
    with open("./tools/annotation_rulemessage/allrule.txt", "r", encoding="utf-8") as f:
        data = f.read()
        data2 = data.replace('‘', '"').replace('’', '"').replace('“', '"').replace('”', '"')
    with open("./tools/annotation_rulemessage/allrule.txt", "w", encoding="utf-8") as p:
        p.writelines(data2)
