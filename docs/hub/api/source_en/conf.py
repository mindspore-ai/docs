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
sys.path.append(os.path.abspath('../_ext'))
from sphinx.ext import autodoc as sphinx_autodoc
from sphinx.util import inspect as sphinx_inspect
from sphinx.domains import python as sphinx_domain_python
from textwrap import dedent

import mindspore_hub

# -- Project information -----------------------------------------------------

project = 'MindSpore'
copyright = '2020, MindSpore'
author = 'MindSpore'

# The full version, including alpha/beta/rc tags
release = 'master'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
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
exclude_patterns = []

pygments_style = 'sphinx'

autodoc_inherit_docstrings = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# -- Options for Texinfo output -------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', '../python_objects.inv'),
    'numpy': ('https://docs.scipy.org/doc/numpy/', '../numpy_objects.inv'),
}

# Modify default signatures for autodoc.
autodoc_source_path = os.path.abspath(sphinx_autodoc.__file__)
inspect_source_path = os.path.abspath(sphinx_inspect.__file__)
autodoc_source_re = re.compile(r"(\s+)args = self\.format_args\(\*\*kwargs\)")
inspect_source_code_str = """signature = inspect.signature(subject)"""
inspect_target_code_str = """signature = my_signature.signature(subject)"""
autodoc_source_code_str = """args = self.format_args(**kwargs)"""
is_autodoc_code_str = """args = args.replace("'", "")"""
with open(autodoc_source_path, "r+", encoding="utf8") as f:
    code_str = f.read()
    if is_autodoc_code_str not in code_str:
        code_str_lines = code_str.split("\n")
        autodoc_target_code_str = None
        for line in code_str_lines:
            re_matched_str = autodoc_source_re.search(line)
            if re_matched_str:
                space_num = re_matched_str.group(1)
                autodoc_target_code_str = dedent("""\
                    {0}
                    {1}if type(args) != type(None):
                    {1}    {2}""".format(autodoc_source_code_str, space_num, is_autodoc_code_str))
                break
        if autodoc_target_code_str:
            code_str = code_str.replace(autodoc_source_code_str, autodoc_target_code_str)
            exec(code_str, sphinx_autodoc.__dict__)
with open(inspect_source_path, "r+", encoding="utf8") as g:
    code_str = g.read()
    if inspect_target_code_str not in code_str:
        code_str = code_str.replace(inspect_source_code_str, inspect_target_code_str)
        if "import my_signature" not in code_str:
            code_str = code_str.replace("import sys", "import sys\nimport my_signature")
        exec(code_str, sphinx_inspect.__dict__)

# remove extra space for default params for autodoc.
sphinx_domain_python_source_path = os.path.abspath(sphinx_domain_python.__file__)
python_code_source = """for argument in arglist.split(','):"""
python_code_target = """for argument in [" " + i if num > 1 else i for num,i in enumerate(arglist.split(", "))]:"""
with open(sphinx_domain_python_source_path, "r+", encoding="utf8") as f:
    code_str = f.read()
    if python_code_target not in code_str:
        code_str = code_str.replace(python_code_source, python_code_target)
        exec(code_str, sphinx_domain_python.__dict__)


from sphinx.search import jssplitter as sphinx_split
from sphinx import errors as searchtools_path

# Update the word segmentation method, let the input term be segmented according to the index
sphinx_split_python = os.path.abspath(sphinx_split.__file__) # Read the location of the word segmentation file
python_code_source = """function splitQuery(query, dict, all_dict) {
    var result = [];
    var tmp = []
    for (var i = 0; i < dict.length; i++) {
        if (query.indexOf(dict[i])!=-1) {
          tmp.push(dict[i])
        }
    }
    if (escape(query).indexOf("%u")== -1 && query.indexOf(all_dict[i])==-1){
      query = query.split('.').slice(-1)
      return query
    }
    if (!tmp.length){
      return [query]
    }
    min_freq = all_dict[tmp[0]].length
    var min_freq_word = tmp[0]
    for (var i = 0; i < tmp.length-1; i++) {
        var a = all_dict[tmp[i]].length
        if (a<min_freq){
          min_freq = a
          min_freq_word = tmp[i]
        }
    }
    result.push(min_freq_word)
    return result;
}"""
python_code_target = """function splitQuery(query) {
    var result = [];
    var start = -1;
    for (var i = 0; i < query.length; i++) {
        if (splitChars[query.charCodeAt(i)]) {
            if (start !== -1) {
                result.push(query.slice(start, i));
                start = -1;
            }
        } else if (start === -1) {
            start = i;
        }
    }
    if (start !== -1) {
        result.push(query.slice(start));
    }
    return result;
}"""
with open(sphinx_split_python, "r+", encoding="utf8") as f:
    code_str = f.read()
    if python_code_target in code_str:
        code_str = code_str.replace(python_code_target,python_code_source)
        f.seek(0)
        f.truncate()
        f.write(code_str)


# Update sphinx searchtools,Read the word segmentation index
sphinx_search_prepare = os.path.abspath(searchtools_path.__file__).replace("errors.py","")+"themes/basic/static/searchtools.js"
dict_target =""" var terms = this._index.terms;
    var titleterms = this._index.titleterms;"""
dict_source = """ """

# Get the index of entries containing Chinese
search_prepare_source = """
    var terms = this._index.terms
    var titleterms = this._index.titleterms
    //Make a deep copy of the original dictionary
    var distinct_dict = Object.assign({},terms)
    //Combine two indexes and eliminate duplicates
    for (key in titleterms){
      if (key in distinct_dict && distinct_dict[key] instanceof Array){
        distinct_dict[key]=distinct_dict[key].concat(titleterms[key])
      }
      else if((key in distinct_dict && !(distinct_dict[key] instanceof Array) )){
        distinct_dict[key]=[]
        distinct_dict[key]=distinct_dict[key].concat(terms[key])
        distinct_dict[key]=distinct_dict[key].concat(titleterms[key])
      }
      else{
        distinct_dict[key]=titleterms[key]
      }
    }
    //Take out items with Chinese characters in the dictionary
    chinese_dic = []
    var dic = []
    dic = Object.keys(distinct_dict)
    for (i=0;i<dic.length;i++){
      if (escape(dic[i]).indexOf("%u")>=0 && dic[i].length>=2){
        chinese_dic.push(dic[i]);
      }    
    }
    var tmp = splitQuery(query, chinese_dic, distinct_dict);"""
search_prepare_target = """var tmp = splitQuery(query);"""
with open(sphinx_search_prepare, "r+", encoding="utf8") as f:
    code_str = f.read()
    if dict_target in code_str:
        code_str = code_str.replace(dict_target,dict_source)
        f.seek(0)
        f.truncate()
        f.write(code_str)

    if search_prepare_target in code_str:
        code_str = code_str.replace(search_prepare_target,search_prepare_source)
        f.seek(0)
        f.truncate()
        f.write(code_str)
