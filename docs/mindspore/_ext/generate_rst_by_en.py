"""Automatically generate Chinese ops documents based on English."""
import importlib
import inspect
import re
import os
from functools import reduce

def get_api(fullname):
    """
    获取接口对象。

    :param fullname: 接口名全称
    :return: 属性对象或None(如果不存在)
    """
    main_module = fullname.split('.')[0]
    main_import = importlib.import_module(main_module)

    try:
        return reduce(getattr, fullname.split('.')[1:], main_import)
    except AttributeError:
        return None

def generate_rst_by_en(sum_list, target_path, language='cn'):
    """Generate the rst file by the ops list."""

    exist_rst = []
    primi_auto = []
    for i in sum_list:
        # 处理大写接口
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

        # 获取入参
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
            # 记录哪些是自动生成的
            if py_docs:
                primi_auto.append(i)
            # 生成中文rst文件
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
    return exist_rst, primi_auto
