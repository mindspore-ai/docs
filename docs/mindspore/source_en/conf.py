# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import re
import sys
import regex
import sphinx
import shutil
import IPython
sys.path.append(os.path.abspath('../_ext'))
import sphinx.ext.autosummary.generate as g
from sphinx.ext import autodoc as sphinx_autodoc

# Fix some dl-label lack class='simple'
from docutils.writers import _html_base

with open(_html_base.__file__, "r", encoding="utf-8") as f:
    code_str = f.read()
    old_str = '''        if self.is_compactable(node):
            classes.append('simple')'''
    new_str = '''        if classes == []:
            classes.append('simple')'''
    code_str = code_str.replace(old_str, new_str)
    exec(code_str, _html_base.__dict__)

# Adjust the display of definition names for Tensor overloaded functions
from sphinx.domains import python as domain_py
with open(domain_py.__file__, 'r', encoding="utf8") as f:
    code_str = f.read()
    old_str = "signode += addnodes.desc_addname(nodetext, nodetext)"
    new_str = """signode += addnodes.desc_addname(nodetext, nodetext)
        elif 'Tensor' == classname:
            signode += addnodes.desc_addname('Tensor.', 'Tensor.')"""
    code_str = code_str.replace(old_str, new_str)
    exec(code_str, domain_py.__dict__)

# Fix mathjax tags
from sphinx.ext import mathjax as sphinx_mathjax

with open(sphinx_mathjax.__file__, "r", encoding="utf-8") as f:
    code_str = f.read()
    old_str = r'''        if r'\\' in part:
            self.body.append(r'\begin{split}' + part + r'\end{split}')'''
    new_str = r'''        if r'\\' in part:
            if r'\tag{' in part:
                part1, part2 = part.split(r'\tag{')
                self.body.append(r'\begin{split}' + part1 + r'\end{split}' + r'\tag{' +part2)
            else:
                self.body.append(r'\begin{split}' + part + r'\end{split}')'''
    code_str = code_str.replace(old_str, new_str)
    exec(code_str, sphinx_mathjax.__dict__)

from sphinx.ext import viewcode
with open('../_ext/overwriteviewcode_en.txt', 'r', encoding="utf8") as f:
    exec(f.read(), viewcode.__dict__)

with open('../_ext/overwriteautosummary_generate.txt', 'r', encoding="utf8") as f:
    exec(f.read(), g.__dict__)

# -- Project information -----------------------------------------------------

project = 'MindSpore'
copyright = 'MindSpore'
author = 'MindSpore'

# The full version, including alpha/beta/rc tags
release = 'master'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
myst_enable_extensions = ["dollarmath", "amsmath"]


myst_heading_anchors = 5
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinxcontrib.mermaid',
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.mathjax',
    'IPython.sphinxext.ipython_console_highlighting'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
mathjax_path = 'https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/mathjax/MathJax-3.2.2/es5/tex-mml-chtml.js'

mathjax_options = {
    'async':'async'
}

nbsphinx_requirejs_path = 'https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.6/require.min.js'

nbsphinx_requirejs_options = {
    "crossorigin": "anonymous",
    "integrity": "sha256-1fEPhSsRKlFKGfK3eO710tEweHh1fwokU5wFGDHO+vg="
}

smartquotes_action = 'De'

exclude_patterns = []

pygments_style = 'sphinx'

autodoc_inherit_docstrings = False

autosummary_generate = True

autosummary_generate_overwrite = False

html_static_path = ['_static']

mermaid_version = ""

mermaid_init_js = ""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

import sphinx_rtd_theme
layout_target = os.path.join(os.path.dirname(sphinx_rtd_theme.__file__), 'layout.html')
layout_src = '../../../resource/_static/layout.html'
if os.path.exists(layout_target):
    os.remove(layout_target)
shutil.copy(layout_src, layout_target)

# -- Options for Texinfo output -------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', '../../../resource/python_objects.inv'),
    'numpy': ('https://docs.scipy.org/doc/numpy/', '../../../resource/numpy_objects.inv'),
}

# Modify default signatures for autodoc.
autodoc_source_path = '../_ext/overwrite_autodoc.txt'
autodoc_source_re = re.compile(r'stringify_signature\(.*?\)')
get_param_func_str = r"""\
import re
import inspect as inspect_

def get_param_func(func):
    try:
        source_code = inspect_.getsource(func)
        if func.__doc__:
            source_code = source_code.replace(func.__doc__, '')
        all_params_str = re.findall(r"def [\w_\d\-]+\(([\S\s]*?)(\):|\) ->.*?:)", source_code)
        if "@classmethod" in source_code:
            all_params = re.sub("(self|cls)(, |,)?", '', all_params_str[0][0].replace("\n", ""))
            if ',' in all_params_str[0][0]:
                all_params = re.sub("(self|cls)(, |,)", '', all_params_str[0][0].replace("\n", ""))
        elif "def __new__" in source_code:
            all_params = re.sub("(self|cls|value|iterable)(, |,)?", '', all_params_str[0][0].replace("\n", ""))
            if ',' in all_params:
                all_params = re.sub("(self|cls|value|iterable)(, |,)", '', all_params_str[0][0].replace("\n", ""))
        else:
            all_params = re.sub("(self)(, |,)?", '', all_params_str[0][0].replace("\n", ""))
            if ',' in all_params_str[0][0]:
                all_params = re.sub("(self)(, |,)", '', all_params_str[0][0].replace("\n", ""))
        return all_params
    except:
        return ''

def get_obj(obj):
    if isinstance(obj, type):
        try:
            test_source = inspect_.getsource(obj.__init__)
        except:
            return obj.__new__
        return obj.__init__

    return obj
"""

with open(autodoc_source_path, "r", encoding="utf8") as f:
    code_str = f.read()
    code_str = autodoc_source_re.sub('"(" + get_param_func(get_obj(self.object)) + ")"', code_str, count=0)
    exec(get_param_func_str, sphinx_autodoc.__dict__)
    exec(code_str, sphinx_autodoc.__dict__)

# Repair error decorators defined in mindspore.
try:
    decorator_list = [("mindspore/common/_decorator.py", "deprecated",
                       "    def decorate(func):",
                       "    def decorate(func):\n\n        import functools\n\n        @functools.wraps(func)"),
                       ("mindspore/nn/optim/optimizer.py", "deprecated",
                       "def opt_init_args_register(fn):\n    \"\"\"Register optimizer init args.\"\"\"\n",
                       "def opt_init_args_register(fn):\n    \"\"\"Register optimizer init args.\"\"\"\n\n    import functools\n\n    @functools.wraps(fn)"),
                       ("mindspore/log.py", "deprecated",
                       "    def __call__(self, func):\n",
                       "    def __call__(self, func):\n        import functools\n\n        @functools.wraps(func)\n"),
                       ("mindspore/ops/primitive.py", "fix for `shard`",
                       "    @_LogActionOnce(logger=logger, key='Primitive')", "    # The decorator has been deleted."),
                       ("mindspore/dataset/engine/datasets.py","generate api",
                       "    @deprecated(\"1.5\")","    # The decorator has been deleted(id1)."),
                       ("mindspore/dataset/engine/datasets.py","generate api",
                       "    @check_bucket_batch_by_length","    # The decorator has been deleted(id2)."),
                       ("mindspore/train/summary/summary_record.py", "summary_record",
                       "            value (Union[Tensor, GraphProto, TrainLineage, EvaluationLineage, DatasetGraph, UserDefinedInfo,\n                LossLandscape]): The value to store.\n\n", 
                       "            value (Union[Tensor, GraphProto, TrainLineage, EvaluationLineage, DatasetGraph, UserDefinedInfo, LossLandscape]): The value to store.\n\n"),
                       ("mindspore/nn/cell.py","generate api",
                       "    @jit_forbidden_register","    # generate api by del decorator."),
                       ("mindspore/profiler/dynamic_profiler.py","generate api",
                       "    @no_exception_func()","    # generate api by del decorator.")]

    base_path = os.path.dirname(os.path.dirname(sphinx.__file__))
    for i in decorator_list:
        with open(os.path.join(base_path, os.path.normpath(i[0])), "r+", encoding="utf8") as f:
            content = f.read()
            if i[3] not in content:
                content = content.replace(i[2], i[3])
                f.seek(0)
                f.truncate()
                f.write(content)
except:
    pass

# Repair error content defined in mindspore.
try:
    decorator_list = [("mindspore/common/dtype.py","del decorator",
                       "@enum.unique","# generate api by del decorator."),
                      ("mindspore/common/dtype.py","del class",
                       "class QuantDtype(enum.Enum):","class QuantDtype():")]

    base_path = os.path.dirname(os.path.dirname(sphinx.__file__))
    for i in decorator_list:
        with open(os.path.join(base_path, os.path.normpath(i[0])), "r+", encoding="utf8") as f:
            content = f.read()
            if i[2] in content:
                content = content.replace(i[2], i[3])
                f.seek(0)
                f.truncate()
                f.write(content)
except:
    pass

# add @functools.wraps
try:
    decorator_list = [("mindspore/common/_tensor_overload.py", ".*?_mint")]

    base_path = os.path.dirname(os.path.dirname(sphinx.__file__))
    for i in decorator_list:
        with open(os.path.join(base_path, os.path.normpath(i[0])), "r+", encoding="utf8") as f:
            content = f.read()
            new_content = re.sub('(import .*\n)', r'\1import functools\n', content, 1)
            new_content = re.sub(f'def ({i[1]})\((.*?)\):\n((?:.|\n|)+?)([ ]+?)def wrapper\(',
                             rf'def \1(\2):\n\3\4@functools.wraps(\2)\n\4def wrapper(', new_content)
            if new_content != content:
                f.seek(0)
                f.truncate()
                f.write(new_content)
except:
    pass

sys.path.append(os.path.abspath('../../../resource/search'))
import search_code

re_url = r"(((gitee.com/mindspore/docs)|(github.com/mindspore-ai/(mindspore|docs))|" + \
         r"(mindspore.cn/(docs|tutorials|lite))|(obs.dualstack.cn-north-4.myhuaweicloud)|" + \
         r"(mindspore-website.obs.cn-north-4.myhuaweicloud))[\w\d/_.-]*?)/(master)"

re_url2 = r"(gitee.com/mindspore/mindspore[\w\d/_.-]*?)/(master)"

re_url4 = r"(((gitee.com/mindspore/mindformers)|(mindspore.cn/mindformers))[\w\d/_.-]*?)/(dev)"

base_path = os.path.dirname(os.path.dirname(sphinx.__file__))
for cur, _, files in os.walk(os.path.join(base_path, 'mindspore')):
    for i in files:
        if i.endswith('.py'):
            with open(os.path.join(cur, i), 'r+', encoding='utf-8') as f:
                content = f.read()
                new_content = re.sub(re_url, r'\1/r2.6.0rc1', content)
                new_content = re.sub(re_url2, r'\1/v2.6.0-rc1', new_content)
                new_content = re.sub(re_url4, r'\1/r1.5.0', new_content)
                if new_content != content:
                    f.seek(0)
                    f.truncate()
                    f.write(new_content)

# Copy source files of en python api from mindspore repository.
copy_path = 'docs/api/api_python_en'
repo_path = os.getenv("MS_PATH")
src_dir_en = os.path.join(repo_path, copy_path)

des_sir = "./api_python"

def copy_source(sourcedir, des_sir):
    for i in os.listdir(sourcedir):
        if os.path.isfile(os.path.join(sourcedir, i)):
            if os.path.exists(os.path.join(des_sir, i)):
                os.remove(os.path.join(des_sir, i))
            shutil.copy(os.path.join(sourcedir, i), os.path.join(des_sir, i))
        else:
            if os.path.exists(os.path.join(des_sir, i)):
                shutil.rmtree(os.path.join(des_sir, i))
            shutil.copytree(os.path.join(sourcedir, i), os.path.join(des_sir, i))

no_viewsource_list = [os.path.join(des_sir,i) for i in os.listdir(des_sir)]

copy_source(src_dir_en, des_sir)

Tensor_list_path = "./api_python/Tensor_list.rst"
dataset_list_path = "./api_python/dataset_list.rst"
classtemplate_path = "./_templates/classtemplate.rst"
classtemplate_dataset_path = "./_templates/classtemplate_dataset.rst"
if os.path.exists(classtemplate_path):
    os.remove(classtemplate_path)
if os.path.exists(classtemplate_dataset_path):
    os.remove(classtemplate_dataset_path)
shutil.copy(Tensor_list_path, classtemplate_path)
shutil.copy(dataset_list_path, classtemplate_dataset_path)
if os.path.exists(Tensor_list_path):
    os.remove(Tensor_list_path)
if os.path.exists(dataset_list_path):
    os.remove(dataset_list_path)

def ops_interface_name():

    src_target_path = os.path.join(src_dir_en, 'mindspore.ops.primitive.rst')
    with open(src_target_path,'r',encoding='utf8') as f:
        content =  f.read()
    primi_list = re.findall("    (mindspore\.ops\.\w*?)\n", content)

    return primi_list

def mint_interface_name():
    mint_p = 'mindspore.mint.rst'
    src_target_path = os.path.join(src_dir_en, mint_p)
    with open(src_target_path,'r',encoding='utf8') as f:
        content =  f.read()
    mint_list = re.findall(r"    (mindspore\.mint\..*)\n", content+'\n')

    return mint_list

try:
    primitive_list = ops_interface_name()
except:
    pass

mint_sum = mint_interface_name()

# auto generate rst by en
from generate_rst_by_en import generate_rst_by_en

exist_rst_file, primi_auto = generate_rst_by_en(primitive_list, './api_python/ops', language='en')

# Rename .rst file to .txt file for include directive.
from rename_include import rename_include

rename_include('migration_guide')

# modify urls
import json

if os.path.exists('../../../tools/generate_html/version.json'):
    with open('../../../tools/generate_html/version.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)
elif os.path.exists('../../../tools/generate_html/daily_dev.json'):
    with open('../../../tools/generate_html/daily_dev.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)
elif os.path.exists('../../../tools/generate_html/daily.json'):
    with open('../../../tools/generate_html/daily.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)

if repo_path.split('/')[-1]:
    copy_repo = repo_path.split('/')[-1]
else:
    copy_repo = repo_path.split('/')[-2]

branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == copy_repo][0]
docs_branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == 'tutorials'][0]
repo_whl = 'mindspore/python/'
giturl = 'https://gitee.com/mindspore/'
ops_yaml = 'mindspore/ops/op_def/yaml/doc/'
tensor_yaml = 'mindspore/ops/api_def/method_doc/'
func_yaml = 'mindspore/ops/api_def/function_doc/'

try:
    ops_yaml_list = [i for i in os.listdir(os.path.join(repo_path, 'mindspore/ops/op_def/yaml/doc')) if i.endswith('_doc.yaml') and '_grad' not in i]
except:
    ops_yaml_list = []

func_name_dict = {}
for i in os.listdir(os.path.join(repo_path, 'mindspore/ops/op_def/yaml')):
    if i.endswith('_op.yaml') and '_grad' not in i:
        with open(os.path.join(repo_path, 'mindspore/ops/op_def/yaml', i), 'r+', encoding='utf-8') as f:
            op_content = f.read()
            if re.findall('function:\n\s+?name: (.*)', op_content):
                func_name_dict[re.findall('function:\n\s+?name: (.*)', op_content)[0]] = i.replace('_op.yaml', '')

for cur, _, files in os.walk(des_sir):
    for i in files:
        if os.path.join(cur, i) in no_viewsource_list:
            continue
        if i.endswith('.rst') or i.endswith('.md') or i.endswith('.ipynb'):
            with open(os.path.join(cur, i), 'r+', encoding='utf-8') as f:
                content = f.read()
                new_content = re.sub(re_url, r'\1/r2.6.0rc1', content)
                new_content = re.sub(re_url4, r'\1/r1.5.0', new_content)
                if i.endswith('.rst'):
                    new_content = re.sub(re_url2, r'\1/v2.6.0-rc1', new_content)
                # if i.endswith('.md'):
                #     md_view = f'[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/{copy_repo}/blob/{branch}/' + copy_path + cur.split('api_python')[-1] + '/' + i + ')\n\n'
                #     if 'resource/_static/logo_source' not in new_content:
                #         new_content = re.sub('(# .*\n\n)', r'\1'+ md_view, new_content, 1)
                if new_content != content:
                    f.seek(0)
                    f.truncate()
                    f.write(new_content)

import mindspore

from myautosummary import MsPlatformAutoSummary, MsNoteAutoSummary, MsPlatWarnAutoSummary

sys.path.append(os.path.abspath('../../../resource/custom_directives'))
from custom_directives import IncludeCodeDirective

def setup(app):
    app.add_directive('msplatformautosummary', MsPlatformAutoSummary)
    app.add_directive('msplatwarnautosummary', MsPlatWarnAutoSummary)
    app.add_directive('msnoteautosummary', MsNoteAutoSummary)
    app.add_directive('includecode', IncludeCodeDirective)
    app.add_js_file('js/mermaid-9.3.0.js')
    app.add_config_value('docs_branch', '', True)
    app.add_config_value('branch', '', True)
    app.add_config_value('copy_repo', '', True)
    app.add_config_value('giturl', '', True)
    app.add_config_value('repo_whl', '', True)
    app.add_config_value('ops_yaml', '', True)
    app.add_config_value('tensor_yaml', '', True)
    app.add_config_value('func_yaml', '', True)
    app.add_config_value('ops_yaml_list', [], True)
    app.add_config_value('primi_auto', [], True)
    app.add_config_value('func_name_dict', {}, True)
    app.add_config_value('mint_sum', [], True)
    app.add_config_value('repo_path', '', True)

# Copy images from mindspore repo.
import imghdr
from sphinx.util import logging

logger = logging.getLogger(__name__)
src_dir = os.path.join(repo_path, 'docs/api/api_python')
des_dir = "./api_python"
image_specified = {"train/": ""}

if not os.path.exists(src_dir):
    logger.warning(f"不存在目录：{src_dir}！")

def copy_image(sourcedir, des_dir):
    """
    Copy all images from sourcedir to workdir.
    """
    for cur, _, files in os.walk(sourcedir, topdown=True):
        for i in files:
            if imghdr.what(os.path.join(cur, i)):
                try:
                    rel_path = os.path.relpath(cur, sourcedir)
                    targetdir = os.path.join(des_dir, rel_path)
                    for j in image_specified.keys():
                        if rel_path.startswith(j):
                            value = image_specified[j]
                            targetdir = os.path.join(des_dir,  re.sub(rf'^{j}', rf'{value}', rel_path))
                            break
                    if not os.path.exists(targetdir):
                        os.makedirs(targetdir, exist_ok=True)
                    shutil.copy(os.path.join(cur, i), targetdir)
                except:
                    logger.warning(f'picture {os.path.join(os.path.relpath(cur, sourcedir), i)} copy failed.')

copy_image(src_dir, des_dir)

# src_release = os.path.join(repo_path, 'RELEASE.md')
# des_release = "./RELEASE.md"
# release_source = f'[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/{copy_repo}/blob/{branch}/' + 'RELEASE.md)\n'

# with open(src_release, "r", encoding="utf-8") as f:
#     data = f.read()

# hide_release = []
# if len(re.findall("\n## (.*?)\n",data)) > 1:
#     for i in hide_release:
#         del_doc = re.findall(f"(\n## MindSpore {i}[\s\S\n]*?)\n## ", data)
#         if del_doc:
#             data = data.replace(del_doc[0], '')
#     content = regex.findall("(\n## MindSpore [^L][\s\S\n]*?)\n## ", data, overlapped=True)
#     repo_version = re.findall("\n## MindSpore ([0-9]+?\.[0-9]+?)\.([0-9]+?)[ -]", content[0])[0]
#     content_new = ''
#     for i in content:
#         if re.findall(f"\n## MindSpore ({repo_version[0]}\.[0-9]+?)[ -]", i):
#             content_new += i
#     content = content_new
# else:
#     content = re.findall("(\n## [\s\S\n]*)", data)
#     content = content[0]

# with open(des_release, "w", encoding="utf-8") as p:
#     content = re.sub(re_url, r'\1/r2.6.0rc1', content)
#     content = re.sub(re_url2, r'\1/v2.6.0-rc1', content)
#     p.write("# Release Notes" + "\n\n" + release_source)
#     p.write(content)
