"""check notebook"""
import json
import re
import subprocess
import sys
import os


def py_rule_generator():
    """生成py检测规则到本地"""
    rule_lines = [
        '[MASTER]\n', '\n', 'profile=no\n', 'ignore=CVS\n', 'persistent=yes\n', 'load-plugins=\n', '\n', '\n',
        '[MESSAGES CONTROL]\n', '\n', 'enable=indexing-exception,old-raise-syntax\n', '\n',
        'disable=design,similarities,no-self-use,attribute-defined-outside-init,locally-disabled,star-args,\
        pointless-except,bad-option-value,global-statement,fixme,suppressed-message,useless-suppression,\
        locally-enabled,no-member,no-name-in-module,import-error,unsubscriptable-object,\
        unbalanced-tuple-unpacking,undefined-variable,not-context-manager\n', '\n', 'cache-size=500\n', '\n',
        '\n', '[REPORTS]\n', '\n', 'output-format=text\n', '\n', 'files-output=no\n', '\n', 'reports=no\n',
        '\n', 'evaluation=10.0 - ((float(5 * error + warning + refactor + convention) / statement) * 10)\n',
        '\n', 'comment=no\n', '\n', '\n', '[TYPECHECK]\n', '\n', 'ignore-mixin-members=yes\n', '\n',
        'ignored-classes=SQLObject\n', '\n', 'zope=no\n', '\n',
        'generated-members=REQUEST,acl_users,aq_parent\n', '\n',
        'contextmanager-decorators=contextlib.contextmanager,contextlib2.contextmanager\n', '\n', '\n',
        '[VARIABLES]\n', '\n', 'init-import=no\n', '\n', 'dummy-variables-rgx=^\\*{0,2}(_$|unused_|dummy_)\n',
        '\n', 'additional-builtins=\n', '\n', '\n', '[BASIC]\n', '\n', 'required-attributes=\n', '\n',
        'bad-functions=apply,input,reduce\n', '\n', '\n', 'disable-report=\n', '\n',
        'module-rgx=(([a-z_][a-z0-9_]*)|([A-Z][a-zA-Z0-9]+))$\n', '\n',
        'const-rgx=^(_?[A-Z][A-Z0-9_]*|__[a-z0-9_]+__|_?[a-z][a-z0-9_]*)$\n',
        '\n', 'class-rgx=^_?[A-Z][a-zA-Z0-9]*$\n', '\n',
        'function-rgx=^(?:(?P<camel_case>_?[A-Z][a-zA-Z0-9]*)|(?P<snake_case>_?[a-z][a-z0-9_]*))$\n',
        '\n', 'method-rgx=^(?:(?P<exempt>__[a-z0-9_]+__|next)|(?P<camel_case>_{0,2}[A-Z][a-zA-Z0-9]*)\
        |(?P<snake_case>_{0,2}[a-z][a-z0-9_]*))$\n', '\n',
        'attr-rgx=^_{0,2}[a-z][a-z0-9_]*$\n', '\n', 'argument-rgx=^[a-z][a-z0-9_]*$\n',
        '\n', 'variable-rgx=^[a-z][a-z0-9_]*$\n', '\n',
        'class-attribute-rgx=^(_?[A-Z][A-Z0-9_]*|__[a-z0-9_]+__|_?[a-z][a-z0-9_]*)$\n',
        '\n', 'inlinevar-rgx=^[a-z][a-z0-9_]*$\n', '\n', 'good-names=main,_\n',
        '\n', 'bad-names=\n', '\n', 'no-docstring-rgx=(__.*__|main)\n', '\n', 'docstring-min-length=10\n',
        '\n', '\n', '[FORMAT]\n', '\n', 'max-line-length=120\n',
        '\n', 'ignore-long-lines=(?x)\n', '  (^\\s*(import|from)\\s\n', '   |\\$Id:\\s\\/\\/depot\\/.+\n',
        '   |^[a-zA-Z_][a-zA-Z0-9_]*\\s*=\\s*("[^"]\\S+"|\'[^\']\\S+\')\n', '   |^\\s*\\#\\ LINT\\.ThenChange\n',
        '   |^[^#]*\\#\\ type:\\ [a-zA-Z_][a-zA-Z0-9_.,[\\] ]*$\n', '   |pylint\n', '   |"""\n', '   |\\\n',
        '   |lambda\n', '   |(https?|ftp):)\n', '\n', 'single-line-if-stmt=y\n', '\n', 'no-space-check=\n', '\n',
        'max-module-lines=99999\n', '\n', "indent-string='    '\n", '\n', '\n', '[SIMILARITIES]\n', '\n',
        'min-similarity-lines=4\n', '\n', 'ignore-comments=yes\n', '\n', 'ignore-docstrings=yes\n', '\n',
        'ignore-imports=no\n', '\n', '\n', '[MISCELLANEOUS]\n', '\n', 'notes=\n', '\n', '\n', '[IMPORTS]\n', '\n',
        'deprecated-modules=regsub,TERMIOS,Bastion,rexec,sets\n', '\n', 'import-graph=\n', '\n',
        'ext-import-graph=\n', '\n', 'int-import-graph=\n', '\n', '\n', '[CLASSES]\n', '\n',
        'ignore-iface-methods=isImplementedBy,deferred,extends,names,namesAndDescriptions,queryDescriptionFor,\
        getBases,getDescriptionFor,getDoc,getName,getTaggedValue,getTaggedValueTags,isEqualOrExtendedBy,\
        setTaggedValue,isImplementedByInstancesOf,adaptWith,is_implemented_by\n',
        '\n', 'defining-attr-methods=__init__,__new__,setUp\n', '\n', 'valid-classmethod-first-arg=cls,class_\n',
        '\n', 'valid-metaclass-classmethod-first-arg=mcs\n', '\n', '\n', '[DESIGN]\n', '\n', 'max-args=5\n',
        '\n', 'ignored-argument-names=_.*\n', '\n', 'max-locals=15\n', '\n', 'max-returns=6\n', '\n',
        'max-branches=12\n', '\n', 'max-statements=50\n', '\n', 'max-parents=7\n', '\n', 'max-attributes=7\n',
        '\n', 'min-public-methods=2\n', '\n', 'max-public-methods=20\n', '\n', '\n', '[EXCEPTIONS]\n', '\n',
        'overgeneral-exceptions=Exception,StandardError,BaseException\n', '\n', '\n', '[AST]\n', '\n',
        'short-func-length=1\n', '\n', 'deprecated-members=string.atof,string.atoi,string.atol,string.capitalize,\
        string.expandtabs,string.find,string.rfind,string.index,string.rindex,string.count,string.lower,\
        string.split,string.rsplit,string.splitfields,string.join,string.joinfields,string.lstrip,string.rstrip,\
        string.strip,string.swapcase,string.translate,string.upper,string.ljust,string.rjust,string.center,\
        string.zfill,string.replace,sys.exitfunc\n', '\n', '\n', '[DOCSTRING]\n',
        '\n', 'ignore-exceptions=AssertionError,NotImplementedError,StopIteration,TypeError\n',
        '\n', '\n', '\n', '[TOKENS]\n', '\n', 'indent-after-paren=4\n', '\n', '\n', '[MINDSPORE LINES]\n',
        '\n', 'copyright=Copyright \\d{4} The MindSpore Authors\\. +All [Rr]ights [Rr]eserved\\.'
    ]
    with open("./pylintrc", "w", encoding="utf-8") as f:
        f.writelines(rule_lines)

def md_rule_generator():
    """"生成markdown检测规则到本地"""
    rule_lines = [
        "all\n",
        "rule 'MD007', :indent => 4\n",
        "rule 'MD009', :br_spaces => 2\n",
        "rule 'MD029', :style => :ordered\n",
        "exclude_rule 'MD013'\n",
        "exclude_rule 'MD002'\n",
        "exclude_rule 'MD041'\n",
        "exclude_rule 'MD005'\n",
        "exclude_rule 'MD024'\n",
        "exclude_rule 'MD033'\n",
        "exclude_rule 'MD029'\n",
        "exclude_rule 'MD034'\n"
    ]
    with open("./mdrules.rb", "w", encoding="utf-8") as f:
        f.writelines(rule_lines)

def get_notebook_cells(path):
    """获取ipynb文档的cells"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["cells"]

def check_notebook(path):
    """检测空白单元格"""
    cells = get_notebook_cells(path)
    for num, cell in enumerate(cells):
        if not cell["source"]:
            print("{}:cell_{}:empty cell need to delete".format(path, num+1))

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
            if not cell["source"]:
                continue
            k_name = "cell_{}".format(num+1)
            if cell["source"][-1].endswith("\n"):
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
            if not cell["source"]:
                continue
            if cell["source"][-1].endswith("\n"):
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
    check_command = "mdl -s mdrules.rb" if lint_name == "md" else "pylint -j 4"
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
    error_code = ""
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
            error_code = re.findall(r"([A-Z]+\d+)", error_info)[0]
            infos = "{}:{}:{}:{} \"{}\"".format(file, cell_num, line, error_info, detail_content)
        except IndexError:
            pass
        finally:
            pass
        yield infos, error_code

def white_code():
    """过滤报错信息"""
    white = [
        'MD034', 'MD029', 'MD033', 'MD024', 'MD005', 'MD041', 'MD002', 'MD013',
        'C0111', 'C0103', 'C0301', 'C0413', 'C0412', 'C0411', 'C0114', 'W0404',
        'C0116', 'C0115', 'E0102', 'E1003', 'E1120', 'E1123', 'R1720', 'R1721',
        'R1723', 'R1725', 'R1732', 'W0106', 'W0221', 'W0613', 'W0622', 'W0632',
        'E1129', 'E0602', 'E1136', 'E0401', 'E0611', 'E1101', 'W0511', 'W0603',
        'W0201', 'R0201', 'R0801', 'R0902', 'R0913', 'R0901', 'R0916', 'R0914',
        'R0904', 'R0911', 'R0915', 'W0621',
    ]
    ignore_code = []
    for arg in sys.argv[1:]:
        if arg.startswith("--ignore="):
            ignore_code = arg.split("=")[-1].split(",")
    white += ignore_code
    return set(white)

def run(path):
    """运行检测"""
    check_notebook(path)
    checkinfo_1 = check_lint(path, lint_name="md")
    loc_1 = location(path)
    print_information_1 = print_info(checkinfo_1, loc_1)

    checkinfo_2 = check_lint(path, lint_name="py")
    loc_2 = location_py(path)
    print_information_2 = print_info(checkinfo_2, loc_2)

    for i, error_code in print_information_2:
        if error_code not in white_code():
            print(i)
    for i, error_code in print_information_1:
        if error_code not in white_code():
            print(i)


if __name__ == "__main__":
    md_rule_generator()
    py_rule_generator()
    for check_path_ in sys.argv[1:]:
        if os.path.isfile(check_path_):
            run(check_path_)
        elif os.path.isdir(check_path_):
            ipynb_f = [file for file in find_file(check_path_, files=[]) if file.endswith(".ipynb")]
            for one_ipynb_f in ipynb_f:
                run(one_ipynb_f)
    if os.path.exists("./mdrules.rb"):
        os.remove("./mdrules.rb")
    if os.path.exists("./pylintrc"):
        os.remove("./pylintrc")
