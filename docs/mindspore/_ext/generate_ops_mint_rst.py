"""Generate mint interface documentation that is consistent with the ops interface content."""
import os
import re
import shutil
import importlib
# pylint: disable=W0611
import mindspore
from mindspore.mint import optim

def generate_ops_mint_rst(repo_path, ops_path, mint_path, pr_need='all'):
    """Generate mint interface documentation that is consistent with the ops interface content."""
    if not pr_need:
        return 1

    mint_ops_dict = dict()

    with open(os.path.join(repo_path, 'docs/api/api_python/mindspore.mint.rst'), 'r+', encoding='utf-8') as f:
        content = f.read()
    sum_list = re.findall(r'(mindspore\.mint\.[\w.]+?)\n', content+'\n')

    # pylint: disable=R1702
    # pylint: disable=w0612
    for root, dirs, files in os.walk(os.path.join(repo_path, 'mindspore/python/mindspore/mint')):
        for file in files:
            if file in ('__init__.py', 'functional.py'):
                with open(os.path.join(root, file), 'r+', encoding='utf-8') as f:
                    content = f.read()
                modulename = '.'.join(root.split('mindspore/python/')[-1].split('/'))
                if file == 'functional.py':
                    modulename += '.functional'
                mint_ops_dict[modulename] = []
                # pylint: disable=eval-used
                try:
                    module = importlib.import_module(modulename)
                    reg_all = getattr(module, '__all__')
                # pylint: disable=W0702
                except:
                    print(f'{modulename}没有__all__属性或没有{modulename}模块')
                    continue
                one_p = re.findall(r'from mindspore\.(ops|nn).*?(?<!extend) import (.*?)(\n|# )', content)
                two_p = [i[1] for i in one_p]
                for i in two_p:
                    if ' as ' in i:
                        name1 = re.findall('(.*?) as (.*)', i)[0][0]
                        name2 = re.findall('(.*?) as (.*)', i)[0][1]
                        if name1 != name2 + '_ext' and name1 != name2 + 'Ext' and name2 in reg_all:
                            mint_ops_dict[modulename].append([name1, name2])
                        else:
                            continue
                    elif i in reg_all:
                        mint_ops_dict[modulename].append(i)
                    else:
                        for j in i.split(','):
                            if j.strip() in reg_all:
                                mint_ops_dict[modulename].append(j.strip())

    exist_mint_file = []
    print('已自动生成与ops内容一致的mint接口:')
    if pr_need != 'all':
        os.makedirs(mint_path, exist_ok=True)
    for k, v in mint_ops_dict.items():
        if not re.findall(r'\.nn\.(?!function).*', k):
            for name in v:
                if isinstance(name, list):
                    name1 = name[0]
                    name2 = name[1]
                else:
                    name1 = name
                    name2 = name
                if f'{k}.{name2}' not in sum_list:
                    continue
                elif pr_need != 'all' and f'{k}.{name2}' not in pr_need:
                    continue
                if os.path.exists(os.path.join(mint_path, f'{k}.{name2}.rst')):
                    exist_mint_file.append(f'{k}.{name2}')
                    continue
                if os.path.exists(os.path.join(ops_path, f'mindspore.ops.func_{name1}.rst')):
                    shutil.copy(os.path.join(ops_path, f'mindspore.ops.func_{name1}.rst'),
                                os.path.join(mint_path, f'{k}.{name2}.rst'))
                    with open(os.path.join(mint_path, f'{k}.{name2}.rst'), 'r+', encoding='utf-8') as f:
                        content = f.read()
                        content = content.replace(f'mindspore.ops.{name1}', f'{k}.{name2}')
                        need_equal = len(f'{k}.{name2}') - len(f'mindspore.ops.{name1}')
                        content = re.sub(rf'{k}.{name2}\n([=]+)', rf'{k}.{name2}\n\1'+'='*need_equal, content)
                        # 同步英文的替换部分(非支持平台、样例部分)
                        content = re.sub(r'(:(func|class):`[^`]+?)ops\.([^`]+?` 的别名)', r'\1mint.\3', content)
                        content = re.sub(r'(\n[ ]+(-|\*) GPU(/CPU)?(:|：).*\n|\n[ ]+(-|\*) CPU(/GPU)?(:|：).*\n)',
                                         r'\n', content)
                        f.seek(0)
                        f.truncate()
                        f.write(content)
                    print(f'{k}.{name2}')
    print('已存在mint中文文档，未覆盖成功的如下：')
    for i in exist_mint_file:
        print(i)

    return 1
