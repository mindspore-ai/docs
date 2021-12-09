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
"""search code"""
import os
from sphinx.search import jssplitter as sphinx_split
from sphinx import errors as searchtools_path

# Update the word segmentation method, let the input term be segmented according to the index
sphinx_split_python = os.path.abspath(sphinx_split.__file__) # Read the location of the word segmentation file
python_code_source = """function splitQuery(query, dict, all_dict) {
    var result = [];
    if (query.includes(" ")) {
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
    } else {
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
    }
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
        code_str = code_str.replace(python_code_target, python_code_source)
        f.seek(0)
        f.truncate()
        f.write(code_str)


# Update sphinx searchtools,Read the word segmentation index
sphinx_search_prepare = os.path.abspath(searchtools_path.__file__).replace("errors.py", "") + \
"themes/basic/static/searchtools.js"
dict_target = """ var terms = this._index.terms;
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

sort_results_target = """var resultCount = results.length;"""

sort_results_source = r"""
    var lists = results;
    var resultCount = lists.length;
    function sortItem() {
      for (i = 0; i < results.length; i++) {
        var requestUrl = "";
        if (DOCUMENTATION_OPTIONS.BUILDER === 'dirhtml') {
          // dirhtml builder
          var dirname = results[i][0] + '/';
          if (dirname.match(/\/index\/$/)) {
            dirname = dirname.substring(0, dirname.length-6);
          } else if (dirname == 'index/') {
            dirname = '';
          }
          requestUrl = DOCUMENTATION_OPTIONS.URL_ROOT + dirname;

        } else {
          // normal html builders
          requestUrl = DOCUMENTATION_OPTIONS.URL_ROOT + results[i][0] + DOCUMENTATION_OPTIONS.FILE_SUFFIX;
        }
        if (results[i][3]){
          results[i].push(true);
        } else if (DOCUMENTATION_OPTIONS.HAS_SOURCE) {
          $.ajax({url: requestUrl,
            dataType: "text",
            async: false,
            complete: function(jqxhr, textstatus) {
              var data = jqxhr.responseText;
              if (data !== '' && data !== undefined) {
                if (Search.makeSearchSummary(data, [query], hlterms).length) {
                  results[i].push(true);
                  results[i].push(Search.makeSearchSummary(data, [query], hlterms));
                } else {
                  results[i].push(false);
                  results[i].push(Search.makeSearchSummary(data, searchterms, hlterms));
                }
              }
            }});
        }
      }
      beforList = [];
      afterList = [];
      for (i = 0; i < results.length; i++) {
        if (results[i][6]) {
          beforList.push(results[i]);
        } else {
          afterList.push(results[i]);
        }
      }
      results = afterList.concat(beforList);
    }
    if (resultCount && resultCount < 100) {
      sortItem()
    };"""

results_function_target = """$.ajax({url: requestUrl,
                  dataType: "text",
                  complete: function(jqxhr, textstatus) {
                    var data = jqxhr.responseText;
                    if (data !== '' && data !== undefined) {
                      listItem.append(Search.makeSearchSummary(data, searchterms, hlterms));
                    }
                    Search.output.append(listItem);
                    listItem.slideDown(5, function() {
                      displayNextItem();
                    });
                  }});"""

results_function_source = """if (resultCount < 100) {
          listItem.append(item[7]);
          Search.output.append(listItem);
          listItem.slideDown(5, function() {
            displayNextItem();
          });
          } else {
          $.ajax({url: requestUrl,
                  dataType: "text",
                  complete: function(jqxhr, textstatus) {
                    var data = jqxhr.responseText;
                    if (data !== '' && data !== undefined) {
                      if (Search.makeSearchSummary(data, [query], hlterms).length) {
                        listItem.append(Search.makeSearchSummary(data, [query], hlterms));
                      } else {
                        listItem.append(Search.makeSearchSummary(data, searchterms, hlterms));
                      }
                    }
                    Search.output.append(listItem);
                    listItem.slideDown(5, function() {
                      displayNextItem();
                    });
                  }});
          };"""

highlight_words_target = """start = Math.max(start - 120, 0);"""

highlight_words_source = """if (start === 0) {
      return [];
    }
    var number = Math.max(start - 120, 0);
    start = number;"""

with open(sphinx_search_prepare, "r+", encoding="utf8") as f:
    code_str = f.read()
    if dict_target in code_str:
        code_str = code_str.replace(dict_target, dict_source)
        f.seek(0)
        f.truncate()
        f.write(code_str)

    if search_prepare_target in code_str:
        code_str = code_str.replace(search_prepare_target, search_prepare_source)
        f.seek(0)
        f.truncate()
        f.write(code_str)

    if sort_results_target in code_str:
        code_str = code_str.replace(sort_results_target, sort_results_source)
        f.seek(0)
        f.truncate()
        f.write(code_str)

    if results_function_target in code_str:
        code_str = code_str.replace(results_function_target, results_function_source)
        f.seek(0)
        f.truncate()
        f.write(code_str)

    if highlight_words_target in code_str:
        code_str = code_str.replace(highlight_words_target, highlight_words_source)
        f.seek(0)
        f.truncate()
        f.write(code_str)
