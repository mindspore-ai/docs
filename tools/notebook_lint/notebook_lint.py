"""check notebook"""
import json
import re
import subprocess
import sys
import os


def get_notebook_cells(path):
    """获取ipynb文档的cells"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cells"]

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

def location(path):
    """构建xxx_lint.md坐标的坐标"""
    location_dict = {}
    line_num_start = 1
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for num, cell in enumerate(data["cells"]):
        if cell["cell_type"] == "code":
            continue
        else:
            k_name = "cell_{}".format(num+1)
            if cell["source"][-1] == "\n":
                line_num_end = line_num_start + len(cell["source"]) + 1
            else:
                line_num_end = line_num_start + len(cell["source"])
            location_dict[k_name] = "{}_{}".format(line_num_start, line_num_end)
            line_num_start = line_num_end + 1
    return location_dict

def convert_to_markdown(path):
    """从ipynb文件中提取markdown的source内容"""
    content = []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for cell in data["cells"]:
        if cell["cell_type"] == "markdown":
            if content:
                content += ["\n"]
            content.extend(cell["source"]+["\n"])
    save_path = path.replace(".ipynb", "_lint.md")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("".join(content))

def location_py(path):
    """构建xxx_lint.py文档坐标"""
    location_dict = {}
    line_num_start = 1
    cells = get_notebook_cells(path)
    for num, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            k_name = "cell_{}".format(num+1)
            if cell["source"][-1] == "\n":
                line_num_end = line_num_start + len(cell["source"]) + 1
            else:
                line_num_end = line_num_start + len(cell["source"])
            location_dict[k_name] = "{}_{}".format(line_num_start, line_num_end)
            line_num_start = line_num_end + 1
        else:
            continue
    return location_dict

def check_lint(path, lint_name="md"):
    """lint命令及检测信息提取，同时删除中间文档xxx_lint.md或者xxx_lint.py"""
    error_infos = []
    lint_ext = "_lint.md" if lint_name == "md" else "_lint.py"
    check_command = "mdl -s markdownlint_docs.rb" if lint_name == "md" else "pylint"
    if lint_name == "md":
        convert_to_markdown(path)
    if lint_name == "py":
        convert_to_py(path)
    check_path = path.replace(".ipynb", lint_ext)
    cmd = "{} {}".format(check_command, check_path)
    res = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",)
    info = res.stdout.read()
    if info is not None:
        info_list = info.split("\n")[:-1]
        for i in info_list:
            try:
                location_, error_info = re.findall(":([0-9]+):(.*)", i)[0]
                file_name = check_path.replace(lint_ext, ".ipynb")
                error_infos.append((file_name, location_, error_info.strip()))
            except IndexError:
                pass
            finally:
                pass

    os.remove(check_path)
    return error_infos

def convert_to_py(path):
    """从ipynb文档中提取代码内容"""
    content = []
    cells = get_notebook_cells(path)
    for cell in cells:
        if cell["cell_type"] == "code":
            if content:
                content += ["\n"]
            for i in cell["source"]:
                if i.strip().startswith("!") or i.strip().startswith("%"):
                    content.append("# " + i)
                else:
                    content.append(i)
            content.append("\n")

    save_path = path.replace(".ipynb", "_lint.py")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("".join(content))

def print_info(check_info, location_info):
    """生成检测报错信息"""
    infos = ""
    for file, error_line, error_info in check_info:
        cells = get_notebook_cells(file)
        try:
            for k, v in location_info.items():
                if (int(error_line) - int(v.split("_")[0])) * (int(error_line) - int(v.split("_")[1])) <= 0:
                    cell_num = k
                    cell_count = int(k.split("_")[-1])
                    line = int(error_line) - int(v.split("_")[0]) + 1
                    break
            detail_content = cells[cell_count-1]["source"][line-1]
            if len(detail_content) > 30:
                detail_content = detail_content[:30] + "..."
            infos = "{}:{}:{}:{} \"{}\"".format(file, cell_num, line, error_info, detail_content)
        except IndexError:
            pass
        finally:
            pass
        yield infos

def run(path):
    """运行检测"""
    checkinfo_1 = check_lint(path, lint_name="md")
    loc_1 = location(path)
    print_information_1 = print_info(checkinfo_1, loc_1)

    checkinfo_2 = check_lint(path, lint_name="py")
    loc_2 = location_py(path)
    print_information_2 = print_info(checkinfo_2, loc_2)

    for i in print_information_2:
        print(i)
    for i in print_information_1:
        print(i)

if __name__ == "__main__":
    for check_path_ in sys.argv[1:]:
        if os.path.isfile(check_path_):
            run(check_path_)
        elif os.path.isdir(check_path_):
            ipynb_f = [file for file in find_file(check_path_, files=[]) if file.endswith(".ipynb")]
            for one_ipynb_f in ipynb_f:
                run(one_ipynb_f)
