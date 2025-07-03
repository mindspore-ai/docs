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
    try:
        api = getattr(module_import, api_name, '')
    except AttributeError:
        print(f'failed to {module_import}.{api_name}')
        return ''
    return api

def generate_rst_by_en(sum_list, target_path, language='cn'):
    """Generate the rst file by the ops list."""

    exist_rst = []
    primi_auto = []
    for i in sum_list:
        if i.lower() == i:
            continue
        try:
            module_api = get_api(i)
        # pylint: disable=W0702
        except:
            continue
        if not module_api:
            continue
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
            if py_docs and language == 'cn':
                sig_doc_str = all_params.strip()
                cn_base_rst = i + '\n' + '=' * len(i) + '\n\n' + '.. py:class:: ' + i + '(' + sig_doc_str + ')\n\n'
                py_docs_indent = ''
                for j in py_docs.split('\n'):
                    if j != '' and j.count(' ') != len(j):
                        py_docs_indent += '    ' + j + '\n'
                    else:
                        py_docs_indent += '\n'

                all_rst_content = cn_base_rst + \
                                  py_docs_indent.replace('is equivalent to', '等价于')\
                                  .replace('Refer to', '更多详情请查看：').replace('for more details.', '。')
                mint_rp = re.findall(rf':func:`[^`]+?\.ops\.(([^`]+?)(?<!_ext)(`|_ext`))', all_rst_content)
                if mint_rp and target_path.endswith('mint'):
                    b_name = i.split('.')[-1]
                    usename = i.replace('mindspore.', '')
                    all_rst_content = re.sub(rf'ops\.{b_name}(Ext)?', usename, all_rst_content)
                    old_rp = mint_rp[0][0].replace('`', '')
                    new_rp = mint_rp[0][1]
                    if 'mindspore.mint.nn.functional.'+new_rp in sum_list:
                        all_rst_content = all_rst_content.replace(f'ops.{old_rp}', 'mint.nn.functional.'+new_rp)
                    elif 'mindspore.mint.'+new_rp in sum_list:
                        all_rst_content = all_rst_content.replace(f'ops.{old_rp}', 'mint.'+new_rp)
                if os.path.exists(os.path.join(target_path, i + '.rst')):
                    exist_rst.append(i)
                with open(os.path.join(target_path, i + '.rst'), "w", encoding='utf-8') as f:
                    f.write(all_rst_content)
            elif py_docs and language == 'en':
                primi_auto.append(i)
    return exist_rst, primi_auto
