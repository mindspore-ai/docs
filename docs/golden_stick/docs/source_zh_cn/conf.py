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
import glob
import os
import re
import shutil
import sys
from sphinx.ext import autodoc as sphinx_autodoc
import sphinx.ext.autosummary.generate as g

sys.path.append(os.path.abspath('../_ext'))

# -- Project information -----------------------------------------------------

project = 'MindSpore Golden Stick'
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

html_search_language = 'zh'

html_search_options = {'dict': '../../../resource/jieba.txt'}

# -- Options for HTML output -------------------------------------------------

# Reconstruction of sphinx auto generated document translation.

language = 'zh_CN'
locale_dirs = ['../../../../resource/locale/']
gettext_compact = False

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

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', '../../../../resource/python_objects.inv'),
    'numpy': ('https://docs.scipy.org/doc/numpy/', '../../../../resource/numpy_objects.inv'),
}

# Modify regex for sphinx.ext.autosummary.generate.find_autosummary_in_lines.
gfile_abs_path = os.path.abspath(g.__file__)
autosummary_re_line_old = r"autosummary_re = re.compile(r'^(\s*)\.\.\s+autosummary::\s*')"
autosummary_re_line_new = r"autosummary_re = re.compile(r'^(\s*)\.\.\s+(ms[a-z]*)?autosummary::\s*')"
with open(gfile_abs_path, "r+", encoding="utf8") as f:
    data = f.read()
    data = data.replace(autosummary_re_line_old, autosummary_re_line_new)
    exec(data, g.__dict__)

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
        if "@classmethod" in source_code:
            all_params = re.sub("(self|cls)(,|, )?", '', all_params_str[0][0].replace("\n", ""))
        else:
            all_params = re.sub("(self)(,|, )?", '', all_params_str[0][0].replace("\n", ""))
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

from sphinx import directives
with open('../_ext/overwriteobjectiondirective.txt', 'r', encoding="utf8") as f:
    exec(f.read(), directives.__dict__)

from sphinx.ext import viewcode
with open('../_ext/overwriteviewcode.txt', 'r', encoding="utf8") as f:
    exec(f.read(), viewcode.__dict__)

with open("../_ext/customdocumenter.txt", "r", encoding="utf8") as f:
    code_str = f.read()
    exec(code_str, sphinx_autodoc.__dict__)

from myautosummary import MsCnAutoSummary

def setup(app):
    app.add_directive('mscnautosummary', MsCnAutoSummary)
    app.add_config_value('rst_files', set(), False)

# Copy source files of chinese python api from golden-stick repository.
from sphinx.util import logging
import shutil
logger = logging.getLogger(__name__)

copy_path = 'docs/api/api_zh_cn'
src_dir_api = os.path.join(os.getenv("GS_PATH"), copy_path)

copy_list = []
moment_dir=os.path.dirname(__file__)

outer_dir = []
for i in os.listdir(src_dir_api):
    if not os.path.isfile(os.path.join(src_dir_api, i)):
        outer_dir.append(os.path.join(src_dir_api, i))

for root,dirs,files in os.walk(src_dir_api):
    for file in files:
        if root==src_dir_api:
            if os.path.exists(os.path.join(moment_dir, file)):
                os.remove(os.path.join(moment_dir, file))
            shutil.copy(os.path.join(root, file), os.path.join(moment_dir, file))
            continue
        for i in outer_dir:
            if i in root:
                outer_dir_name = i.split('/')[-1]
                os.makedirs(f'./{outer_dir_name}', exist_ok=True)
                if os.path.exists(os.path.join(f'./{outer_dir_name}', file)):
                    os.remove(os.path.join(f'./{outer_dir_name}', file))
                shutil.copy(os.path.join(root, file), os.path.join(f'./{outer_dir_name}', file))
                break
        else:
            if not os.path.exists('.' + root.split(copy_path)[-1]):
                os.makedirs('.' + root.split(copy_path)[-1])
            if os.path.exists('.' + root.split(copy_path)[-1] + '/'+file):
                os.remove('.' + root.split(copy_path)[-1] + '/'+file)
            shutil.copy(os.path.join(root, file), '.' + root.split(copy_path)[-1]+'/'+file)

readme_path = os.path.join(os.getenv("GS_PATH"), 'README_CN.md')

with open(readme_path, 'r', encoding='utf-8') as f:
    content = f.read()

ind_content = 'MindSpore Golden Stick 文档\n=============================\n'

sc_doc = re.findall('\n## 概述\n((?:.|\n|)+?)\n## ', content)
if sc_doc:
    ind_content += re.sub('!\[(.*?)\]\((.*?)\)', r'.. image:: \2', sc_doc[0])
    ind_content = re.sub('.. image:: docs/.*?/(images/.*)', r'.. image:: ./\1', ind_content)
    ind_content = re.sub('\n\n> (.+)', r'\n\n.. note::\n    \1', ind_content)
    ind_content = re.sub('\n> (.+)', r'\n    \1', ind_content)
    ind_content += "\n\n代码仓地址： <https://gitee.com/mindspore/golden-stick>\n"

gsdocs_image = os.path.join(os.getenv("GS_PATH"), 'docs/zh_cn/images')
if not os.path.exists(os.path.join(moment_dir, 'images')):
    shutil.copytree(gsdocs_image, os.path.join(moment_dir, 'images'))

ind_content += """
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: 安装部署

   install

"""

toctree = []

spec_copy = []
toctree_list = re.findall('<thead>(?:.|\n|)+?</thead>\n[ ]+?<tbody>(?:.|\n|)+?</tbody>', content)
if toctree_list:
    for i in toctree_list:
        toctree_n = re.findall('<th .*<div.*?>(.*?)</div>', i)
        if 'demo' in toctree_n[-1] or 'TBD' in toctree_n[-1] or toctree_n[-1] == '概览' or toctree_n[-1] == '其他':
            continue
        toctree_p = []
        if re.findall('<th .*?<a href="(.*?)"', i) and 'README.' not in re.findall('<th .*?<a href="(.*?)"', i)[0].split('/')[-1]:
            href = re.findall('<th .*?<a href="(.*?)"', i)[0]
            toctree_p.append('/'.join(re.findall('<th .*?<a href="(.*?)"', i)[0].split('/')[:-1])+'/overview')
            docs_p = '/'.join(href.replace('mindspore_gs/', '').split('/')[:-1]) + '/overview.' + href.split('.')[-1]
            spec_copy.append([href, docs_p])
        if re.findall('<td .*?<a href="(.*?)">(.*?)<', i):
            for href, name in re.findall('<td .*?<a href="(.*?)">(.*?)<', i):
                if 'demo' not in name and 'README.' not in href.split('/')[-1]:
                    toctree_p.append('/'.join(href.split('/')[:-1]))
                    docs_p = '/'.join(href.replace('mindspore_gs/', '').split('/')[:-1]) + '.' + href.split('.')[-1]
                    spec_copy.append([href, docs_p])
        toctree.append([toctree_n[-1], toctree_p])

for toc_n, toc_p in toctree:
    ind_content += f'.. toctree::\n   :glob:\n   :maxdepth: 1\n   :caption: {toc_n}\n\n'
    for p in toc_p:
        p_new = p.replace('mindspore_gs/', '')
        ind_content += f'   {p_new}\n'
    ind_content += '\n'

with open(os.path.join(src_dir_api, 'index.rst'), 'r', encoding='utf-8') as f:
    api_ind = f.read()

api_toc = re.findall('.. toctree::(?:.|\n|)+', api_ind)[0]
ind_content += api_toc

ind_content += """
.. toctree::
   :glob:
   :maxdepth: 1
   :caption: RELEASE NOTES

   RELEASE
"""
with open(os.path.join('./index.rst'), 'w', encoding='utf-8') as f:
    f.write(ind_content)

for gs_p, f_p in spec_copy:
    ori_p = os.path.join(os.getenv("GS_PATH"), gs_p)
    target_dir = os.path.join(moment_dir, '/'.join(f_p.split('/')[:-1]))
    os.makedirs(target_dir, exist_ok=True)
    if os.path.exists(os.path.join(moment_dir, f_p)):
        os.remove(os.path.join(moment_dir, f_p))
    shutil.copy(ori_p, os.path.join(moment_dir, f_p))

    with open(os.path.join(moment_dir, f_p), 'r+', encoding='utf-8') as f:
        content = f.read()
        if f_p.endswith('.md'):
            content = re.sub('.*?/README.md.*\n.*\n', '', content)
        elif f_p.endswith('.ipynb'):
            content = re.sub('\n.*\[View English\].*\n.*\n', '\n', content, 1)
        f.seek(0)
        f.truncate()
        f.write(content)

    images_path = '/'.join(ori_p.split('/')[:-1]) + '/images/zh_cn'
    os.makedirs(os.path.join(target_dir, 'images/zh_cn'), exist_ok=True)
    if os.path.exists(images_path):
        for i in os.listdir(images_path):
            if os.path.exists(os.path.join(target_dir, 'images/zh_cn', i)):
                os.remove(os.path.join(target_dir, 'images/zh_cn', i))
            shutil.copy(os.path.join(images_path, i), os.path.join(target_dir, 'images/zh_cn', i))

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

if os.getenv("GS_PATH").split('/')[-1]:
    copy_repo = os.getenv("GS_PATH").split('/')[-1]
else:
    copy_repo = os.getenv("GS_PATH").split('/')[-2]

branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == copy_repo.replace('-', '_')][0]
docs_branch = [version_inf[i]['branch'] for i in range(len(version_inf)) if version_inf[i]['name'] == 'tutorials'][0]

re_view = f"\n.. image:: https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/{docs_branch}/" + \
          f"resource/_static/logo_source.svg\n    :target: https://gitee.com/mindspore/{copy_repo}/blob/{branch}/"

for cur, _, files in os.walk(moment_dir):
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
                            re_view_ = re_view + copy_path + cur.split(moment_dir)[-1] + '/' + i + \
                                       '\n    :alt: 查看源文件\n\n'
                            new_content = re.sub('([=]{5,})\n', r'\1\n' + re_view_, content, 1)
                        if new_content != content:
                            f.seek(0)
                            f.truncate()
                            f.write(new_content)
                except Exception:
                    print(f'打开{i}文件失败')

if not os.path.exists(os.path.join(moment_dir, 'install.md')):
    shutil.copy(os.path.join(os.getenv("GS_PATH"), 'docs/zh_cn/install.md'),
                os.path.join(moment_dir, 'install.md'))
    with open(os.path.join(moment_dir, 'install.md'), 'r+', encoding='utf-8') as f:
        content = f.read()
        content = re.sub('\n\[View English\].*\n', '', content, 1)
        f.seek(0)
        f.truncate()
        f.write(content)

sys.path.append(os.path.abspath('../../../../resource/sphinx_ext'))
# import anchor_mod
import nbsphinx_mod

sys.path.append(os.path.abspath('../../../../resource/search'))
import search_code

src_release = os.path.join(os.getenv("GS_PATH"), 'RELEASE_CN.md')
des_release = "./RELEASE.md"
with open(src_release, "r", encoding="utf-8") as f:
    data = f.read()
if len(re.findall("\n## (.*?)\n",data)) > 1:
    content = re.findall("(## [\s\S\n]*?)\n## ", data)
else:
    content = re.findall("(## [\s\S\n]*)", data)
#result = content[0].replace('# MindSpore', '#', 1)
with open(des_release, "w", encoding="utf-8") as p:
    p.write("# Release Notes"+"\n\n")
    p.write(content[0])