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
import sys
import IPython
import re
import nbsphinx as nbs
from sphinx.search import jssplitter as sphinx_split
from sphinx import errors as searchtools_path

# -- Project information -----------------------------------------------------

project = 'MindSpore'
copyright = '2021, MindSpore'
author = 'MindSpore'

# The full version, including alpha/beta/rc tags
release = 'master'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
myst_enable_extensions = ["dollarmath", "amsmath"]

extensions = [
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
exclude_patterns = []

highlight_language = 'none'

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_search_language = 'zh'

html_search_options = {'dict': '../../../resource/jieba.txt'}

# Remove extra outputs for nbsphinx extension.
nbsphinx_source_re = re.compile(r"(app\.connect\('html-collect-pages', html_collect_pages\))")
nbsphinx_math_re = re.compile(r"(\S.*$)")
mod_path = os.path.abspath(nbs.__file__)
with open(mod_path, "r+", encoding="utf8") as f:
    contents = f.readlines()
    for num, line in enumerate(contents):
        _content_re = nbsphinx_source_re.search(line)
        if _content_re and "#" not in line:
            contents[num] = nbsphinx_source_re.sub(r"# \g<1>", line)
        if "mathjax_config = app.config" in line and "#" not in line:
            contents[num:num+10] = [nbsphinx_math_re.sub(r"# \g<1>", i) for i in contents[num:num+10]]
            break
    exec("".join(contents), nbs.__dict__)


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