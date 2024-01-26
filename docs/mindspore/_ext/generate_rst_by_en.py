"""Automatically generate Chinese ops documents based on English."""
import importlib
import inspect
import re
import os

def get_api(fullname):
    """Get the api module."""
    try:
        module_name, api_name = ".".join(fullname.split('.')[:-1]), fullname.split('.')[-1]
        # pylint: disable=unused-variable
        module_import = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_name, api_name = ".".join(fullname.split('.')[:-2]), ".".join(fullname.split('.')[-2:])
        module_import = importlib.import_module(module_name)
    # pylint: disable=eval-used
    api = eval(f"module_import.{api_name}")
    return api

def generate_rst_by_en(ops_list, target_path):
    """Generate the rst file by the ops list."""

    for i in ops_list:
        if i.lower() == i:
            continue
        module_api = get_api(i)
        try:
            if 'mindspore/ops/auto_generate/' not in inspect.getsourcefile(module_api):
                continue
        # pylint: disable=W0702
        except:
            continue
        try:
            py_docs = inspect.getdoc(module_api)
        except TypeError:
            try:
                py_docs = inspect.getdoc(inspect.getmodule(module_api))
            except TypeError:
                py_docs = ''

        try:
            source_code = inspect.getsource(module_api.__init__)
            if module_api.__doc__:
                source_code = source_code.replace(module_api.__doc__, '')
            all_params_str = re.findall(r"def [\w_\d\-]+\(([\S\s]*?)(\):|\) ->.*?:)", source_code)
            all_params = re.sub("(self)(,|, )?", '', all_params_str[0][0].replace("\n", ""))
        # pylint: disable=W0702
        except:
            all_params = ''

        if 'Refer to' in py_docs.split('\n')[-1] and 'for more details.' in py_docs.split('\n')[-1]:
            if py_docs:
                sig_doc_str = all_params.strip()
                cn_base_rst = i + '\n' + '=' * len(i) + '\n\n' + '.. py:class:: ' + i + '(' +sig_doc_str + ')\n\n'
                py_docs_indent = ''
                for j in py_docs.split('\n'):
                    if j != '' and j.count(' ') != len(j):
                        py_docs_indent += '    ' + j + '\n'
                    else:
                        py_docs_indent += '\n'

                all_rst_content = cn_base_rst + \
                                  py_docs_indent.replace('is equivalent to', '等价于')\
                                  .replace('Refer to', '更多详情请查看：').replace('for more details.', '。')
                with open(os.path.join(target_path, i + '.rst'), "w", encoding='utf-8') as f:
                    f.write(all_rst_content)
