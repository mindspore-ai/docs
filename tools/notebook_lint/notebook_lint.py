"""Check Notebook Format"""
import os
import sys
import re
import json
import subprocess
import nbformat

# pylint: disable=R1710,W0235

def find_files(path):
    """遍历目录path中的所有文件路径"""
    return [os.path.join(root, file) for root, _, files in os.walk(path) for file in files]


class ReadFile():
    """文件处理模块"""
    def __init__(self, path):
        self.path = path
        self.json_object = self._get_json()
        self.cells = self._format_cell_source()
        self.markdown = self._markdown_content()
        self.code = self._code_content()

    def _get_json(self):
        return json.load(open(self.path, "r", encoding="utf-8"))

    def _markdown_content(self):
        res = []
        for cell in self.cells:
            if cell["cell_type"] == "markdown" and cell["source"]:
                cell["source"][-1] = cell["source"][-1]+"\n\n"
                res += cell["source"]
        return "".join(res)

    def _code_content(self):
        res = []
        for cell in self.cells:
            if cell["cell_type"] == "code" and cell["source"]:
                cell["source"][-1] = cell["source"][-1]+"\n\n"
                res += cell["source"]
        if res:
            res[-1] = res[-1].replace("\n\n", "\n")
        return "".join(res)

    def _format_cell_source(self):
        cell_source_content = []
        for cell in self.json_object["cells"]:
            if isinstance(cell["source"], str):
                cell["source"] = [line+"\n" for line in cell["source"].split("\n")]
                cell["source"][-1] = cell["source"][-1].strip("\n")
            cell_source_content.append(cell)
        return cell_source_content

    @staticmethod
    def write_content(fpath, content):
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)


class CustomCheck(ReadFile):
    """自定义检测模块"""
    def __init__(self, path):
        super(CustomCheck, self).__init__(path)

    def line_break_and_blank_cell_check(self):
        """空白单元及单元尾多余换行检测"""
        error_info = []
        error_report = "The end of the line in the source code should end with \\n"
        for cell_num, cell in enumerate(self.cells, 1):
            if not cell["source"]:
                error_info.append((self.path, f"cell_{cell_num}", "--", "CC097", "blank cell need to delete"))
            else:
                for line_num, line in enumerate(cell["source"], 1):
                    if not line.endswith("\n") and line_num != len(cell["source"]):
                        error_info.append((self.path, f"cell_{cell_num}", line_num, "CC098", error_report))
        return error_info

    def check_mathematical_formula(self):
        """检测notebook中的数学公式空行问题"""
        error_info = []
        error_code = "CC099"
        error_report = "Mathematical formulas should be surrounded by blank lines"
        math_f = re.findall(r"\$\$([\S\s]*?)\$\$", self.markdown)
        for i in math_f:
            left = "\n\n$$"+i
            right = i+"$$\n\n"
            left_error = not left.replace(" ", "") in self.markdown.replace(" ", "")
            right_error = not right.replace(" ", "") in self.markdown.replace(" ", "")
            if left_error:
                left_error_line = self.markdown.split(i)[0].count("\n") + 1
                cell_num, line_num = self.position_cell_line(left_error_line)
                error_info.append((self.path, f"cell_{cell_num}", line_num, error_code, error_report))
            if right_error:
                right_error_line = self.markdown.split(i)[0].count("\n") + left.count("\n") - 1
                cell_num, line_num = self.position_cell_line(right_error_line)
                error_info.append((self.path, f"cell_{cell_num}", line_num, error_code, error_report))
        return error_info

    def position_cell_line(self, temp_line, error_type="markdown"):
        """将临时文件中的报错行转换为Jupyter Notebook文档中报错单元中的报错行"""
        cell_num = 0
        for cell in self.cells:
            cell_num += 1
            if cell["cell_type"] != error_type or not cell["source"]:
                continue
            if len(cell["source"]) >= temp_line + 1:
                break
            elif cell["source"][-1].endswith("\n\n\n"):
                temp_line = temp_line - len(cell["source"]) - 2
            else:
                temp_line = temp_line - len(cell["source"]) - 1
            if temp_line <= 0:
                temp_line = temp_line + len(cell["source"])
                break
        return cell_num, temp_line

    def check(self):
        """执行自定义检测"""
        error_info = []
        error_info += self.line_break_and_blank_cell_check()
        error_info += self.check_mathematical_formula()
        return error_info


class Notebook_Pylint(ReadFile):
    """Notebook的Pylint检测模块"""
    def __init__(self, path):
        super(Notebook_Pylint, self).__init__(path)
        self.generate_rules()

    def generate_rules(self):
        """生成Pylint的检测规则"""
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

        rule_except = [
            'profile', 'cache-size', 'files-output', 'comment', 'zope',
            'required-attributes', 'bad-functions', 'disable-report', 'no-space-check',
            'ignore-iface-methods', 'short-func-length',
            'deprecated-members', 'ignore-exceptions', 'copyright'
            ]

        import pylint
        pylint_version = pylint.__version__
        if pylint_version >= "2.14.0":
            rule_lines = [i for i in rule_lines if i.split("=")[0] not in rule_except]
        with open("./pylintrc", "w", encoding="utf-8") as f:
            f.writelines(rule_lines)

    def positioning_cell_line(self, temp_line):
        """将临时文件中的报错行转换为notebook文档中报错单元中的报错行"""
        cell_num = 0
        for cell in self.cells:
            cell_num += 1
            if cell["cell_type"] != "code" or not cell["source"]:
                continue
            if len(cell["source"]) >= temp_line:
                break
            elif cell["source"][-1].endswith("\n\n\n"):
                temp_line = temp_line - len(cell["source"]) - 2
            else:
                temp_line = temp_line - len(cell["source"]) - 1
            if temp_line <= 0:
                temp_line = temp_line + len(cell["source"]) + 1
                break
        return cell_num, temp_line

    def process_msg(self, pl_msg_list):
        """处理Pylint的检测信息"""
        error_info = []
        for msg in pl_msg_list:
            if ":/" in msg or ":\\" in msg:
                msg = msg.replace(":/", "/").replace(":\\", "\\")
            _, line_num, _, error_code, error_report = msg.split(":", 4)
            error_code = error_code.strip()
            error_report = error_report.strip()
            cell_num, line_num = self.positioning_cell_line(int(line_num))
            error_info.append((self.path, f"cell_{cell_num}", line_num, error_code, error_report))
        return error_info

    def check(self):
        """执行Pylint检测"""
        temp_file = self.path[:-6] + "_lint.py"
        self.write_content(temp_file, self.code)
        cmd = ["pylint", "-j", "4", temp_file]
        res = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",)
        info = res.stdout.read()
        msgs = [i for i in info.split("\n") if i][1:-2]
        resutls = self.process_msg(msgs)
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return resutls


class Notebook_Markdownlint(ReadFile):
    """Notebook中的Markdownlint检测模块"""
    def __init__(self, path):
        super(Notebook_Markdownlint, self).__init__(path)
        self.generate_rules()

    def generate_rules(self):
        """生成Markdownlint的检测规则"""
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

    def positioning_cell_line(self, temp_line):
        """将临时文件中的报错行转换为Notebook文档中报错单元中的报错行"""
        cell_num = 0
        for cell in self.cells:
            cell_num += 1
            if cell["cell_type"] != "markdown" or not cell["source"]:
                continue
            if len(cell["source"]) >= temp_line:
                break
            elif cell["source"][-1].endswith("\n\n\n"):
                temp_line = temp_line - len(cell["source"]) - 2
            else:
                temp_line = temp_line - len(cell["source"]) - 1
            if temp_line <= 0:
                temp_line = temp_line + len(cell["source"])
                break
        return cell_num, temp_line

    def process_msg(self, mdl_msg_list):
        """处理Markdownlint的报错信息"""
        error_info = []
        for msg in mdl_msg_list:
            if ":/" in msg or ":\\" in msg:
                msg = msg.replace(":/", "/").replace(":\\", "\\")
            _, line_num, error_content = msg.split(":", 2)
            error_code, error_report = error_content.strip().split(" ", 1)
            cell_num, line_num = self.positioning_cell_line(int(line_num))
            error_info.append((self.path, f"cell_{cell_num}", line_num, error_code, error_report))
        return error_info

    def check(self):
        """执行Markdownlint检测"""
        temp_file = self.path[:-6] + "_lint.md"
        self.write_content(temp_file, self.markdown)
        cmd = ["mdl", "-s", "mdrules.rb", temp_file]
        res = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",)
        info = res.stdout.read()
        msgs = [i for i in info.split("\n") if i][:-1]
        results = self.process_msg(msgs)
        if os.path.exists(temp_file):
            os.remove(temp_file)
        return results

class PrintInfo():
    """检测信息输出模块"""
    def __init__(self, message):
        """message格式应该为[(文件名，报错单元，单元行，错误码，报错信息),...]"""
        self.message = message

    def _white_code_filter(self):
        """白名单white中的错误码信息将会过滤掉"""
        white = [
            'MD034', 'MD029', 'MD033', 'MD024', 'MD005', 'MD041', 'MD002', 'MD013',
            'C0111', 'C0103', 'C0301', 'C0413', 'C0412', 'C0411', 'C0114', 'W0404',
            'C0116', 'C0115', 'E0102', 'E1003', 'E1120', 'E1123', 'R1720', 'R1721',
            'R1723', 'R1725', 'R1732', 'W0106', 'W0221', 'W0613', 'W0622', 'W0632',
            'E1129', 'E0602', 'E1136', 'E0401', 'E0611', 'E1101', 'W0511', 'W0603',
            'W0201', 'R0201', 'R0801', 'R0902', 'R0913', 'R0901', 'R0916', 'R0914',
            'R0904', 'R0911', 'R0915', 'W0621', "W0104", 'W0108'
            ]
        ignore_code = []
        for arg in sys.argv[1:]:
            if arg.startswith("--ignore="):
                ignore_code = arg.split("=")[-1].split(",")
        white += ignore_code
        return [msg for msg in self.message if msg[3] not in white]

    def nblprint(self):
        """打印检测结果"""
        msgs = self._white_code_filter()
        for m in msgs:
            print(":".join(map(str, m)))


def init_check(path):
    """检查Notebook的文档格式是否符合Jupyter Notebook的文档格式标准"""
    try:
        nb = nbformat.read(path, as_version=4)
        format_check = nbformat.validator.iter_validate(nb)
        try:
            val = next(format_check)
            return val
        except StopIteration:
            return None
    except nbformat.reader.NotJSONError as e:
        try:
            with open(path, "r", encoding="utf-8") as f:
                _ = json.load(f)
        except json.decoder.JSONDecodeError as e:
            return f"{path}: JSONDecodeError: {e}"
    return f"{path}: read notebook error"


def run(path):
    """执行单个Jupyter Notebook文件检测的主入口"""
    error_message = []
    infos = init_check(path)
    if not infos:
        cc = CustomCheck(path).check()
        nbm = Notebook_Markdownlint(path).check()
        nbp = Notebook_Pylint(path).check()
        error_message.extend(cc)
        error_message.extend(nbm)
        error_message.extend(nbp)
        PrintInfo(error_message).nblprint()
    else:
        print(infos)

if  __name__ == "__main__":
    for check_path_ in sys.argv[1:]:
        if os.path.isfile(check_path_):
            run(check_path_)
        elif os.path.isdir(check_path_):
            ipynb_f = [file for file in find_files(check_path_) if file.endswith(".ipynb")]
            for one_ipynb_f in ipynb_f:
                run(one_ipynb_f)
    if os.path.exists("./mdrules.rb"):
        os.remove("./mdrules.rb")
    if os.path.exists("./pylintrc"):
        os.remove("./pylintrc")
