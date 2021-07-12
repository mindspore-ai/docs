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
from sphinx.search import jssplitter as sphinx_split
from sphinx import errors as searchtools_path


# -- Project information -----------------------------------------------------

project = 'MindArmour'
copyright = '2021, MindArmour'
author = 'MindArmour'

# The full version, including alpha/beta/rc tags
release = 'master'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_markdown_tables',
    'myst_parser',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_search_language = 'zh'

html_search_options = {'dict': '../../../resource/jieba.txt'}



# Update the word segmentation method, let the input term be segmented according to the index
sphinx_split_python = os.path.abspath(sphinx_split.__file__) # Read the location of the word segmentation file
python_code_source = """function splitQuery(query,dict) {
    var result = [];
    for (var i = 0; i < dict.length; i++) {
      console.log(dict[i])
        if (query.indexOf(dict[i])!=-1) {
          result.push(dict[i])
        }
    }
    result.push(query)
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
    var dic = []
    dic = Object.keys(terms).concat(Object.keys(titleterms))
    chinese_dic = []
    for (i=0;i<dic.length;i++){
      if (escape(dic[i]).indexOf("%u")>=0 && dic[i].length>=2){
        chinese_dic.push(dic[i]);
      }    
    }
    var tmp = splitQuery(query,chinese_dic);"""
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

