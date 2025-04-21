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
import textwrap
import shutil
import glob
import sphinx.ext.autosummary.generate as g
from sphinx.ext import autodoc as sphinx_autodoc

sys.path.append(os.path.abspath('../_ext'))
sys.path.append(os.path.abspath("../_custom"))
from exhale import graph as exh_graph

# -- Project information -----------------------------------------------------

project = 'MindSpore Lite'
copyright = 'MindSpore'
author = 'MindSpore Lite'

# The full version, including alpha/beta/rc tags
release = 'master'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
myst_enable_extensions = ["dollarmath", "amsmath"]


myst_heading_anchors = 5
extensions = [
    'breathe',
    'exhale',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'myst_parser',
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

smartquotes_action = 'De'

exclude_patterns = []

pygments_style = 'sphinx'

autodoc_inherit_docstrings = False

autosummary_generate = True

autosummary_generate_overwrite = False

html_search_language = 'zh'

html_search_options = {'dict': '../../../../resource/jieba.txt'}

sys.path.append(os.path.abspath('../../../../resource/custom_directives'))
from custom_directives import IncludeCodeDirective

# -- Options for HTML output -------------------------------------------------

# Fix some dl-label lack class='simple'
from docutils.writers import _html_base

with open(_html_base.__file__, "r+", encoding="utf-8") as f:
    code_str = f.read()
    old_str = '''        if self.is_compactable(node):
            classes.append('simple')'''
    new_str = '''        if classes == []:
            classes.append('simple')'''
    code_str = code_str.replace(old_str, new_str)
    exec(code_str, _html_base.__dict__)

# Reconstruction of sphinx auto generated document translation.
import sphinx
import shutil
mo_target = os.path.join(os.path.dirname(sphinx.__file__), 'locale/zh_CN/LC_MESSAGES/sphinx.mo')
if os.path.exists(mo_target):
    os.remove(mo_target)
language = 'zh_CN'
locale_dirs = ['../../../../resource/locale/']
gettext_compact = False

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', '../../../../resource/python_objects.inv'),
    'numpy': ('https://docs.scipy.org/doc/numpy/', '../../../../resource/numpy_objects.inv'),
}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

import sphinx_rtd_theme
layout_target = os.path.join(os.path.dirname(sphinx_rtd_theme.__file__), 'layout.html')
layout_src = '../../../../resource/_static/layout.html'
if os.path.exists(layout_target):
    os.remove(layout_target)
shutil.copy(layout_src, layout_target)

from myautosummary import MsPlatformAutoSummary, MsNoteAutoSummary, MsCnAutoSummary, MsCnPlatformAutoSummary, MsCnNoteAutoSummary

# Modify regex for sphinx.ext.autosummary.generate.find_autosummary_in_lines.
gfile_abs_path = os.path.abspath(g.__file__)
autosummary_re_line_old = r"autosummary_re = re.compile(r'^(\s*)\.\.\s+autosummary::\s*')"
autosummary_re_line_new = r"autosummary_re = re.compile(r'^(\s*)\.\.\s+(ms[a-z]*)?autosummary::\s*')"
with open(gfile_abs_path, "r+", encoding="utf8") as f:
    data = f.read()
    data = data.replace(autosummary_re_line_old, autosummary_re_line_new)
    exec(data, g.__dict__)

from sphinx import directives
with open('../_ext/overwriteobjectiondirective.txt', 'r', encoding="utf8") as f:
    exec(f.read(), directives.__dict__)

from sphinx.ext import viewcode
with open('../_ext/overwriteviewcode.txt', 'r', encoding="utf8") as f:
    exec(f.read(), viewcode.__dict__)

# Modify default signatures for autodoc.
autodoc_source_path = os.path.abspath(sphinx_autodoc.__file__)
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
        all_params = re.sub("(self|cls)(,|, )?", '', all_params_str[0][0].replace("\n", ""))
        return all_params
    except:
        return ''

def get_obj(obj):
    if isinstance(obj, type):
        return obj.__init__

    return obj
"""

with open(autodoc_source_path, "r+", encoding="utf8") as f:
    code_str = f.read()
    code_str = autodoc_source_re.sub('"(" + get_param_func(get_obj(self.object)) + ")"', code_str, count=0)
    exec(get_param_func_str, sphinx_autodoc.__dict__)
    exec(code_str, sphinx_autodoc.__dict__)

# Copy source files of chinese python api from mindspore repository.
from sphinx.util import logging
import shutil
logger = logging.getLogger(__name__)

copy_path = 'docs/api/lite_api_python'
src_dir = os.path.join(os.getenv("MS_PATH"), copy_path)

copy_list = []

present_path = os.path.dirname(__file__)

for i in os.listdir(src_dir):
    if os.path.isfile(os.path.join(src_dir,i)):
        if os.path.exists('./'+i):
            os.remove('./'+i)
        shutil.copy(os.path.join(src_dir,i),'./'+i)
        copy_list.append(os.path.join(present_path,i))
    else:
        if os.path.exists('./'+i):
            shutil.rmtree('./'+i)
        shutil.copytree(os.path.join(src_dir,i),'./'+i)
        copy_list.append(os.path.join(present_path,i))

# add view
import json

if os.path.exists('../../../../tools/generate_html/version.json'):
    with open('../../../../tools/generate_html/version.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)
elif os.path.exists('../../../../tools/generate_html/daily_dev.json'):
    with open('../../../../tools/generate_html/daily_dev.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)
elif os.path.exists('../../../../tools/generate_html/daily.json'):
    with open('../../../../tools/generate_html/daily.json', 'r+', encoding='utf-8') as f:
        version_inf = json.load(f)

if os.getenv("MS_PATH").split('/')[-1]:
    copy_repo = os.getenv("MS_PATH").split('/')[-1]
else:
    copy_repo = os.getenv("MS_PATH").split('/')[-2]

branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == copy_repo][0]
docs_branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == 'tutorials'][0]

re_view = f"\n.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/" + \
          f"resource/_static/logo_source.svg\n    :target: https://gitee.com/mindspore/{copy_repo}/blob/{branch}/"

for cur, _, files in os.walk(present_path):
    for i in files:
        flag_copy = 0
        if i.endswith('.rst'):
            for j in copy_list:
                if j in cur:
                    flag_copy = 1
                    break
            if os.path.join(cur, i) in copy_list or flag_copy:
                try:
                    with open(os.path.join(cur, i), 'r+', encoding='utf-8') as f:
                        content = f.read()
                        new_content = content
                        if '.. include::' in content and '.. automodule::' in content:
                            continue
                        if 'autosummary::' not in content and "\n=====" in content:
                            re_view_ = re_view + copy_path + cur.split(present_path)[-1] + '/' + i + \
                                       '\n    :alt: 查看源文件\n\n'
                            new_content = re.sub('([=]{5,})\n', r'\1\n' + re_view_, content, 1)
                        if new_content != content:
                            f.seek(0)
                            f.truncate()
                            f.write(new_content)
                except Exception:
                    print(f'打开{i}文件失败')

re_url = r"(((gitee.com/mindspore/(mindspore|docs))|(github.com/mindspore-ai/(mindspore|docs))|" + \
         r"(mindspore.cn/(docs|tutorials|lite))|(obs.dualstack.cn-north-4.myhuaweicloud)|" + \
         r"(mindspore-website.obs.cn-north-4.myhuaweicloud))[\w\d/_.-]*?)/(master)"
for cur, _, files in os.walk('./mindspore_lite'):
    for i in files:
        if i.endswith('.rst') or i.endswith('.md') or i.endswith('.ipynb'):
            try:
                with open(os.path.join(cur, i), 'r+', encoding='utf-8') as f:
                    content = f.read()
                    new_content = re.sub(re_url, r'\1/br_base', content)
                    if new_content != content:
                        f.seek(0)
                        f.truncate()
                        f.write(new_content)
            except Exception:
                print(f'打开{i}文件失败')

rst_files = set([i.replace('.rst', '') for i in glob.glob('mindspore_lite/*.rst', recursive=True)])

def setup(app):
    app.add_directive('msplatformautosummary', MsPlatformAutoSummary)
    app.add_directive('msnoteautosummary', MsNoteAutoSummary)
    app.add_directive('mscnautosummary', MsCnAutoSummary)
    app.add_directive('mscnplatformautosummary', MsCnPlatformAutoSummary)
    app.add_directive('mscnnoteautosummary', MsCnNoteAutoSummary)
    app.add_config_value('rst_files', set(), False)
    app.add_directive('includecode', IncludeCodeDirective)

import mindspore_lite

sys.path.append(os.path.abspath('../../../../resource/sphinx_ext'))
# import anchor_mod

sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code



# Add configrator for c++ api output.
# Setup the breathe extension
breathe_projects = {
    "My Project": "./doxyoutput/xml"
}

breathe_default_project = "My Project"


def specificationsForKind(kind):
    '''
    For a given input ``kind``, return the list of reStructuredText specifications
    for the associated Breathe directive.
    '''
    # Change the defaults for .. doxygenclass:: and .. doxygenstruct::
    if kind == "class":
        return [
            ":members:",
            # ":protected-members:",
            # ":private-members:"
        ]
    else:
        return []


# Use exhale's utility function to transform `specificationsForKind`
# defined above into something Exhale can use

from exhale import utils

exhale_args = {
    ############################################################################
    # These arguments are required.                                            #
    ############################################################################
    "containmentFolder": "./generate",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Library API",
    "doxygenStripFromPath": "..",
    ############################################################################
    # Suggested optional arguments.                                            #
    ############################################################################
    "createTreeView": True,
    "exhaleExecutesDoxygen": True,
    "exhaleUseDoxyfile": False,
    "verboseBuild": True,
    "exhaleDoxygenStdin": textwrap.dedent("""
        INPUT = ../include
        INPUT_FILTER = "python3 ../lite_api_filter.py"
        EXTRACT_ALL = NO
        FILE_PATTERNS = *.h
        EXCLUDE_PATTERNS = *schema* *third_party*
        HIDE_UNDOC_CLASSES = YES
        HIDE_UNDOC_MEMBERS = YES
        EXCLUDE_SYMBOLS = operator* GVAR*
        WARNINGS = NO
        ENABLE_SECTIONS = DISPLAY_COMPOUND
        WARN_IF_UNDOCUMENTED = NO
    """),
    'contentsDirectives': False,

    ############################################################################
    # HTML Theme specific configurations.                                      #
    ############################################################################
    # Fix broken Sphinx RTD Theme 'Edit on GitHub' links
    # Search for 'Edit on GitHub' on the FAQ:
    #     http://exhale.readthedocs.io/en/latest/faq.html
    "pageLevelConfigMeta": ":gitee_url: https://gitee.com/mindspore/docs",
    ############################################################################
    # Individual page layout example configuration.                            #
    ############################################################################
    # Example of adding contents directives on custom kinds with custom title
    "contentsTitle": "Page Contents",
    "kindsWithContentsDirectives": ["class", "file", "namespace", "struct"],
    # Exclude PIMPL files from class hierarchy tree and namespace pages.
    # "listingExclude": [r".*Impl$"],
    ############################################################################
    # Main library page layout example configuration.                          #
    ############################################################################
    "afterTitleDescription": textwrap.dedent(u'''
        Welcome to the developer reference for the MindSpore C++ API.
    '''),
    # ... required arguments / other configs ...
    "customSpecificationsMapping": utils.makeCustomSpecificationsMapping(
        specificationsForKind
    )
}

# modify source code of exhale.
exh_file = os.path.abspath(exh_graph.__file__)

with open("../_custom/graph", "r", encoding="utf8") as f:
    source_code = f.read()

exec(source_code, exh_graph.__dict__)

# fix error of extra space for C++ API.
from sphinx.writers import html5 as sphinx_writer_html5

with open("../_custom/sphinx_writer_html5", "r", encoding="utf8") as f:
    source_code = f.read()

exec(source_code, sphinx_writer_html5.__dict__)

# fix position of "Return" for C++ API.
from sphinx.builders import html as sphinx_builder_html

with open("../_custom/sphinx_builder_html", "r", encoding="utf8") as f:
    source_code = f.read()

exec(source_code, sphinx_builder_html.__dict__)

fileList = []
for root, dirs, files in os.walk('../include/'):
    for fileObj in files:
        fileList.append(os.path.join(root, fileObj))

for file_name in fileList:
    file_data = ''
    with open(file_name, 'r', encoding='utf-8') as f:
        data = f.read()
        data = re.sub(r'/\*\*([\s\n\S]*?)\*/', '', data)
    with open(file_name, 'w', encoding='utf-8') as p:
        p.write(data)

#Remove "MS_API" in classes.
# files_copyed = glob.glob("../include/*.h")
# for file in files_copyed:
#     with open(file, "r+", encoding="utf8") as f:
#         content = f.read()
#         if "MS_API" in content:
#             content_new = content.replace("MS_API", "")
#             f.seek(0)
#             f.truncate()
#             f.write(content_new)
