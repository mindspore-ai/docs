# -*- coding: utf-8 -*-
"""
Build web pages for API related document PR and implement preview functionality.
"""
import argparse
import copy
import glob
import imghdr
import json
import os
import re
import shutil
import subprocess
import importlib
import sphinx
import requests
from git import Repo

# 使文件夹存在且为空
def flush(dir_p):
    """
    flush dir.
    """
    if os.path.exists(dir_p):
        shutil.rmtree(dir_p)
    os.makedirs(dir_p)

def update_repo(clone_branch, rp_dir_docs):
    """
    update repo.
    """

    # docs工程运行仓克隆更新
    if not os.path.exists(rp_dir_docs):
        os.makedirs(rp_dir_docs, exist_ok=True)
        Repo.clone_from("https://gitee.com/mindspore/docs.git",
                        rp_dir_docs, branch=clone_branch)
    else:
        # Repo(rp_dir).git.execute(["git","pull","origin","master"])
        repo = Repo(rp_dir_docs)
        str1 = repo.git.execute(["git", "branch", "-a"])
        if not re.findall(f'remotes/origin/{clone_branch}', str1):
            if len(clone_branch.split('.')) >= 3:
                clone_branch = '.'.join(clone_branch.split('.')[:-1])
        try:
            str3 = repo.git.execute(["git", "checkout", clone_branch])
            print(str3)
        # pylint: disable=W0702
        except:
            print(f'docs repo does not have the {clone_branch} branch, the default branch is used for build.')
            clone_branch = re.findall(rf'\* (.*)', str1)[0].strip()

    return clone_branch

def copy_source(sourcedir, des_sir, cp_rel_path, fp_list=''):
    """
    Copy designated files from sourcedir to workdir.
    """
    if fp_list:
        for fp in fp_list:
            if not os.path.exists(os.path.split(os.path.join(des_sir, fp.split(cp_rel_path)[-1]))[0]):
                os.makedirs(os.path.split(os.path.join(
                    des_sir, fp.split(cp_rel_path)[-1]))[0], exist_ok=True)
            if not os.path.exists(os.path.join(des_sir, fp.split(cp_rel_path)[-1])):
                shutil.copy(fp, os.path.join(
                    des_sir, fp.split(cp_rel_path)[-1]))
    elif 'source_en' in des_sir:
        # pylint: disable=W0612
        for root, dirs, files in os.walk(sourcedir):
            for file in files:
                try:
                    if root == sourcedir and file.endswith('.rst') and file.startswith('mindspore.'):
                        continue
                    elif root == sourcedir and file == 'mindspore.multiprocessing.md':
                        continue
                    if not os.path.exists(os.path.join(des_sir, root.split(cp_rel_path)[-1])):
                        os.makedirs(os.path.join(
                            des_sir, root.split(cp_rel_path)[-1]), exist_ok=True)
                    shutil.copy(os.path.join(root, file),
                                os.path.join(des_sir, root.split(cp_rel_path)[-1], file))
                # pylint: disable=W0702
                except:
                    continue


def copy_image(sourcedir, des_dir):
    """
    Copy all images from sourcedir to workdir.
    """
    image_specified = {"train/": ""}
    # pylint: disable=R1702
    for cur, _, files in os.walk(sourcedir, topdown=True):
        for i in files:
            if imghdr.what(os.path.join(cur, i)):
                try:
                    rel_path = os.path.relpath(cur, sourcedir)
                    targetdir = os.path.join(des_dir, rel_path)
                    for j in image_specified:
                        if rel_path.startswith(j):
                            value = image_specified[j]
                            targetdir = os.path.join(des_dir, re.sub(rf'^{j}', rf'{value}', rel_path))
                            break
                    if 'dataset' in targetdir and 'dataset_method' in targetdir and 'source_en' in des_dir:
                        targetdir = targetdir.split('dataset_method')[0] + 'dataset_method'
                    if not os.path.exists(targetdir):
                        os.makedirs(targetdir, exist_ok=True)
                    shutil.copy(os.path.join(cur, i), targetdir)
                # pylint: disable=W0702
                except:
                    print(f'picture {os.path.join(os.path.relpath(cur, sourcedir), i)} copy failed.')

def get_all_copy_list(pr_list, rp_n, branch, repo_path, raw_rst_list):
    """
    获取所有需要拷贝的文件。
    """
    file_list = []
    for i in pr_list:
        if i == 'need_auto':
            continue
        if i.endswith('.rst'):
            raw_content = requests.get(raw_rst_list[i]).text
            other_file_path = re.findall('.. include:: (.*?)\n', raw_content)
            for j in other_file_path:
                file_list.append(os.path.join(
                    repo_path, '/'.join(i.split('/')[:-1]), j.rstrip('\r\t')))
        file_list.append(os.path.join(repo_path, i))

    return file_list

def get_api(fullname):
    """
    获取接口对象。
    """
    try:
        module_name, api_name = ".".join(fullname.split('.')[:-1]), fullname.split('.')[-1]
        # pylint: disable=W0612
        module_import = importlib.import_module(module_name)
        # pylint: disable=W0123
        api = eval(f"module_import.{api_name}")
    except ModuleNotFoundError:
        print(f"not find module, api name: {api_name}")
        return False
    except AttributeError:
        try:
            module_name, api_name = ".".join(fullname.split('.')[:-2]), ".".join(fullname.split('.')[-2:])
            # pylint: disable=W0612
            module_import = importlib.import_module(module_name)
            # pylint: disable=W0123
            api = eval(f"module_import.{api_name}")
        # pylint: disable=W0702
        except:
            print(f'error get_api: {fullname}')
            return False
    except: # pylint: disable=W0702
        print(f'error get_api: {fullname}')
        return False
    return api

def get_all_samedefinition(api_obj, fullname):
    """
    返回同定义的接口。
    """
    all_fullname = []

    if 'mindspore.mint.' in fullname:
        ops_fullname = 'mindspore.ops.' + fullname.split('.')[-1]
        ops_obj = get_api(ops_fullname)
        nn_fullname = 'mindspore.nn.' + fullname.split('.')[-1]
        nn_obj = get_api(nn_fullname)
        mint_func_fullname = 'mindspore.mint.nn.functional.' + fullname.split('.')[-1]
        mint_func_obj = get_api(mint_func_fullname)
        if ops_obj and id(ops_obj) == id(api_obj):
            all_fullname.append(ops_fullname)
        elif nn_obj and id(nn_obj) == id(api_obj):
            all_fullname.append(nn_fullname)
        if mint_func_obj and id(mint_func_obj) == id(api_obj):
            all_fullname.append(mint_func_fullname)
    elif 'mindspore.ops.' in fullname:
        mint_fullname = 'mindspore.mint.' + fullname.split('.')[-1]
        mint_obj = get_api(mint_fullname)
        mint_func_fullname = 'mindspore.mint.nn.functional.' + fullname.split('.')[-1]
        mint_func_obj = get_api(mint_func_fullname)
        if mint_obj and id(mint_obj) == id(api_obj):
            all_fullname.append(mint_fullname)
        if mint_func_obj and id(mint_func_obj) == id(api_obj):
            all_fullname.append(mint_func_fullname)
    elif 'mindspore.nn.' in fullname:
        mint_nn_fullname = 'mindspore.mint.nn.' + fullname.split('.')[-1]
        mint_nn_obj = get_api(mint_nn_fullname)
        if mint_nn_obj and id(mint_nn_obj) == id(api_obj):
            all_fullname.append(mint_nn_fullname)

    if fullname in all_fullname:
        all_fullname.remove(fullname)

    return all_fullname

def get_rst_en(en_list, all_samedfn, samedfn_rst):
    """
    生成英文API文档内容。
    """
    generate_api_en_list = []
    diff_name = []
    all_list = []

    for i in en_list:
        samedfn_rst_list = []
        all_list.append(i)
        one_doc = i.split('&&&')[0]
        api_name = re.findall(r'\.\. .*?:: (.*)', one_doc)[0]
        api_name_obj = ''
        if 'mindspore.nn.' in api_name or 'mindspore.ops.' in api_name or 'mindspore.mint.' in api_name:
            api_name_obj = get_api(api_name)
        if api_name_obj:
            samedef_list = get_all_samedefinition(api_name_obj, api_name)
            if samedef_list:
                all_samedfn.add(api_name)
                samedfn_rst_list.append(api_name)
            for name in samedef_list:
                all_list.append(i.replace(api_name, name))
                all_samedfn.add(name)
                samedfn_rst_list.append(name)
            if samedef_list:
                samedfn_rst.append(sorted(samedfn_rst_list))

    for i in all_list:
        one_doc = i.split('&&&')[0]
        api_name = re.findall(r'\.\. .*?:: (.*)', one_doc)[0]
        if api_name not in diff_name:
            diff_name.append(api_name)
        else:
            continue
        two_doc = i.split('&&&')[1]

        if 'nn.probability' in one_doc:
            generate_api_en_list.append(
                [api_name + '.rst', 'nn_probability', '', two_doc])
            continue
        end_docs = ''
        currentmodule = '.'.join(api_name.split('.')[:-1])
        if 'autoclass' in one_doc:
            end_docs = '\n    :exclude-members: infer_value, infer_shape, infer_dtype, \n    :members:'
        elif 'automethod' in one_doc:
            currentmodule = '.'.join(api_name.split('.')[:-2])

        rst_docs = '.. role:: hidden\n    :class: hidden-section\n\n' + \
            f'.. currentmodule:: {currentmodule}\n\n' + \
            api_name + '\n' + '=' * len(api_name) + \
            '\n\n' + one_doc + end_docs

        rel_path = '/'.join(api_name.split('.')[1:-1])
        if not rel_path:
            rel_path = 'mindspore'
        elif len(api_name.split('.')) >= 4 and 'dataset' in one_doc:
            rel_path = '_'.join(api_name.split('.')[1:-1])
            if '.Dataset.' in api_name:
                rel_path = 'dataset/dataset_method'
        elif '.Tensor.' in api_name:
            rel_path = 'mindspore/Tensor'
        elif '.mint.' in api_name:
            rel_path = 'mint'
        generate_api_en_list.append([api_name, rel_path, rst_docs, two_doc])
    return sorted(generate_api_en_list, key=lambda x: x[0], reverse=False)


def yaml_file_handle(yaml_file_list, repo_path, dict1):
    """
    Identify which API interface corresponds to the yaml file.
    """
    generate_interface_list = []
    for yaml_fp in yaml_file_list:
        yaml_file = os.path.basename(yaml_fp)
        if yaml_file.endswith('.yaml') and '_grad_' not in yaml_file:
            # if re.findall('_v[0-9]+_', yaml_file):
            #     yaml_file = re.sub('_v[0-9]+_', '_', yaml_file)
            if dict1['mindspore_yaml'] in yaml_fp:
                op_fp = os.path.join(
                    repo_path, dict1['mindspore_yaml'], yaml_file.replace('_doc.yaml', '_op.yaml'))

                if not os.path.exists(op_fp):
                    continue
                with open(op_fp, 'r', encoding='utf-8') as f:
                    op_content = f.read()
                class_name = re.findall(r'class:\n\s+?name:(.*)', op_content)
                func_name = re.findall(r'function:\n\s+?name:(.*)', op_content)
                mint_flag = 0
                if '_ext_op.yaml' in op_fp:
                    mint_flag = 1
                if re.findall(r'function:\n\s+?disable: True', op_content):
                    if class_name:
                        class_name = class_name[0]
                    else:
                        class_name = ''.join([i.title() for i in yaml_file.split('_')[:-1]])
                    if class_name.endswith('Ext'):
                        class_name = class_name[:-3]
                    if mint_flag:
                        generate_interface_list.append(
                            f'.. autoclass:: mindspore.mint.{class_name.strip()}&&&{yaml_fp}')
                    else:
                        generate_interface_list.append(
                            f'.. autoclass:: mindspore.ops.{class_name.strip()}&&&{yaml_fp}')
                else:
                    if func_name:
                        func_name = func_name[0]
                    else:
                        func_name = yaml_file.replace('_doc.yaml', '').replace('_op.yaml', '')
                    if func_name.endswith('_ext'):
                        func_name = func_name[:-4]
                    if mint_flag:
                        generate_interface_list.append(
                            f'.. autofunction:: mindspore.mint.{func_name.strip()}&&&{yaml_fp}')
                        generate_interface_list.append(
                            f'.. autofunction:: mindspore.mint.nn.functional.{func_name.strip()}&&&{yaml_fp}')
                    else:
                        generate_interface_list.append(
                            f'.. autofunction:: mindspore.ops.{func_name.strip()}&&&{yaml_fp}')
                        generate_interface_list.append(
                            f'.. autofunction:: mindspore.mint.{func_name.strip()}&&&{yaml_fp}')
                        generate_interface_list.append(
                            f'.. autofunction:: mindspore.mint.nn.functional.{func_name.strip()}&&&{yaml_fp}')
            elif dict1['mindspore_Tensor_yaml'] in yaml_fp:
                tensor_op_name = yaml_file.replace('_doc.yaml', '.yaml')
                op_fp = os.path.join(
                    repo_path, '/'.join(dict1['mindspore_Tensor_yaml'].split('/')[:-2]), tensor_op_name)

                if not os.path.exists(op_fp):
                    continue
                tensor_name = tensor_op_name.split('.')[0]
                generate_interface_list.append(
                    f'.. automethod:: mindspore.Tensor.{tensor_name}&&&{yaml_fp}')
            elif dict1['mindspore_function_yaml'] in yaml_fp:
                func_op_name = yaml_file.replace('_doc.yaml', '.yaml')
                op_fp = os.path.join(
                    repo_path, '/'.join(dict1['mindspore_function_yaml'].split('/')[:-2]), func_op_name)

                if not os.path.exists(op_fp):
                    continue
                func_name = func_op_name.split('.')[0]
                generate_interface_list.append(
                    f'.. autofunction:: mindspore.mint.{func_name}&&&{yaml_fp}')
                generate_interface_list.append(
                    f'.. autofunction:: mindspore.mint.nn.functional.{func_name}&&&{yaml_fp}')
                generate_interface_list.append(
                    f'.. autofunction:: mindspore.ops.{func_name}&&&{yaml_fp}')

    return list(set(generate_interface_list))


def en_file_handle(py_file_list, repo_path, dict1):
    """
    Identify which API interfaces have been modified by the py file.
    """

    # 接口模块分类
    module_path_name = [
        ['mindspore/python/mindspore/parallel/nn', 'mindspore.parallel.nn'],
        ['mindspore/python/mindspore/parallel/auto_parallel', 'mindspore.parallel.auto_parallel'],
        ['mindspore/python/mindspore/parallel', 'mindspore.parallel'],
        ['mindspore/python/mindspore/device_context/cpu', 'mindspore.device_context.cpu'],
        ['mindspore/python/mindspore/device_context/gpu', 'mindspore.device_context.gpu'],
        ['mindspore/python/mindspore/device_context/ascend', 'mindspore.device_context.ascend'],
        ['mindspore/python/mindspore/runtime', 'mindspore.runtime'],
        ['mindspore/python/mindspore/rewrite', 'mindspore.rewrite'],
        ['mindspore/python/mindspore/hal', 'mindspore.hal'],
        ['mindspore/python/mindspore/mindrecord', 'mindspore.mindrecord'],
        ['mindspore/python/mindspore/scipy/linalg.py', 'mindspore.scipy.linalg'],
        ['mindspore/python/mindspore/scipy/optimize', 'mindspore.scipy.optimize'],
        ['mindspore/python/mindspore/scipy/fft.py', 'mindspore.scipy.fft'],
        ['mindspore/python/mindspore/scipy', 'mindspore.scipy'],
        ['mindspore/python/mindspore/train', 'mindspore.train'],
        ['mindspore/python/mindspore/boost', 'mindspore.boost'],
        ['mindspore/python/mindspore/mint/distributed', 'mindspore.mint.distributed'],
        ['mindspore/python/mindspore/mint/optim', 'mindspore.mint.optim'],
        ['mindspore/python/mindspore/mint/nn/functional.py', 'mindspore.mint.nn.functional'],
        ['mindspore/python/mindspore/mint/nn', 'mindspore.mint.nn'],
        ['mindspore/python/mindspore/mint', 'mindspore.mint'],
        ['mindspore/python/mindspore/ops', 'mindspore.ops'],
        ['mindspore/python/mindspore/nn/probability/distribution', 'mindspore.nn.probability.distribution'],
        ['mindspore/python/mindspore/nn/probability/bijector', 'mindspore.nn.probability.bijector'],
        ['mindspore/python/mindspore/nn/extend', 'mindspore.nn.extend'],
        ['mindspore/python/mindspore/nn', 'mindspore.nn'],
        ['mindspore/python/mindspore/dataset/vision', 'mindspore.dataset.vision'],
        ['mindspore/python/mindspore/dataset/text', 'mindspore.dataset.text'],
        ['mindspore/python/mindspore/dataset/audio', 'mindspore.dataset.audio'],
        ['mindspore/python/mindspore/dataset/core/config.py', 'mindspore.dataset.config'],
        ['mindspore/python/mindspore/dataset', 'mindspore.dataset'],
        ['mindspore/python/mindspore/communication/comm_func', 'mindspore.communication.comm_func'],
        ['mindspore/python/mindspore/communication', 'mindspore.communication'],
        ['mindspore/python/mindspore/amp.py', 'mindspore.amp'],
        ['mindspore/python/mindspore/numpy/fft', 'mindspore.numpy.fft'],
        ['mindspore/python/mindspore/numpy', 'mindspore.numpy'],
        ['mindspore/python/mindspore/experimental/optim/lr_scheduler.py', 'mindspore.experimental.optim.lr_scheduler'],
        ['mindspore/python/mindspore/experimental/optim', 'mindspore.experimental.optim'],
        ['mindspore/python/mindspore/common/initializer.py', 'mindspore.common.initializer'],
        ['mindspore/python/mindspore/common', 'mindspore']]

    generate_interface_list = []

    # pylint: disable=R1702
    for i in py_file_list:
        repo_py_path = os.path.join(repo_path, dict1['mindspore_py'], i[0])
        print(repo_py_path)

        with open(repo_py_path, 'r+', encoding='utf-8') as f:
            content = f.read()

        interface_doc_dict = {}
        interface_doc = re.findall(
            r'(\nclass|\ndef|\n[ ]+?def) ([^_].+?:|[^_].+?:[ ]+# .*|[^_].+?,\n(?:.|\n|)+?)\n.*?("""(?:.|\n|)+?)"""', content)

        for doc in interface_doc:
            first_p = doc[0]
            sec_p = doc[1]
            third_p = doc[2]
            if re.findall(r'(\nclass|\ndef|\n[ ]+?def) ', sec_p):
                first_p = re.findall(r'(\nclass|\ndef|\n[ ]+?def) (.+)', sec_p)[-1][0]
                sec_p = re.findall(r'(\nclass|\ndef|\n[ ]+?def) (.+)', sec_p)[-1][1]
                if sec_p.startswith('_'):
                    continue
            if first_p.startswith('\ndef'):
                len_doc = third_p.count('\n')
                interface_name = sec_p.split('(')[0]
                func_name = interface_name + '.'
                index = content.find(third_p)
                begin_line = content[:index].count('\n')
                interface_doc_dict[func_name] = [
                    [begin_line, begin_line + len_doc + 1]]
                if not i[1]:
                    for mpn in module_path_name:
                        if mpn[0] in repo_py_path:
                            generate_interface_list.append(
                                '.. autofunction:: ' + mpn[1] + '.' + func_name.replace('.', '') + f'&&&{i[0]}')
                            break

            elif first_p.startswith('\n '):
                len_doc = third_p.count('\n')
                meth_name = ""
                index = content.find(third_p)
                begin_line = content[:index].count('\n')
                try:
                    interface_name = re.findall(
                        r'\nclass ([^_].+?):', content[:index])[-1]
                    interface_name = interface_name.split('(')[0]
                # pylint: disable=W0702
                except:
                    continue
                if interface_name in ('Tensor', 'Dataset'):
                    meth_name = sec_p.split('(')[0]
                if meth_name:
                    interface_doc_dict[interface_name + '.' +
                                       meth_name] = [[begin_line, begin_line + len_doc + 1]]
                elif interface_name in interface_doc_dict:
                    interface_doc_dict[interface_name].append(
                        [begin_line, begin_line + len_doc + 1])
                else:
                    interface_doc_dict[interface_name] = [[begin_line, begin_line + len_doc + 1]]
            else:
                len_doc = third_p.count('\n')
                interface_name = sec_p.split('(')[0]
                if interface_name.endswith(':'):
                    interface_name = interface_name.rstrip(':')
                index = content.find(third_p)
                begin_line = content[:index].count('\n')
                interface_doc_dict[interface_name] = [
                    [begin_line, begin_line + len_doc + 1]]
                if not i[1]:
                    for mpn in module_path_name:
                        if mpn[0] in repo_py_path:
                            generate_interface_list.append(
                                '.. autoclass:: ' + mpn[1] + '.' + interface_name + f'&&&{i[0]}')
                            break

        if not i[1]:
            continue

        print(i[1])
        # pylint: disable=R1702
        for pr_lines in i[1]:
            # fit = 0
            for k, v in interface_doc_dict.items():
                # if fit:
                #     break
                for py_lines in v:
                    if pr_lines[0] > py_lines[1] or pr_lines[1] < py_lines[0]:
                        continue
                    else:
                        # fit = 1
                        for mpn in module_path_name:
                            if mpn[0] in repo_py_path:
                                if k.endswith('.'):
                                    fullname = mpn[1] + '.' + k.replace('.', '')
                                    if fullname.endswith('_ext') and mpn[1] == 'mindspore.ops':
                                        new_fullname = 'mindspore.mint.' + fullname[:-4].split('.')[-1]
                                        generate_interface_list.append(
                                            '.. autofunction:: ' + new_fullname + f'&&&{i[0]}')
                                        new_fullname = 'mindspore.mint.nn.functional.' + fullname[:-4].split('.')[-1]
                                        generate_interface_list.append(
                                            '.. autofunction:: ' + new_fullname + f'&&&{i[0]}')
                                    else:
                                        generate_interface_list.append(
                                            '.. autofunction:: ' + fullname + f'&&&{i[0]}')
                                elif '.' in k:
                                    generate_interface_list.append(
                                        '.. automethod:: ' + mpn[1] + '.' + k + f'&&&{i[0]}')
                                else:
                                    fullname = mpn[1] + '.' + k
                                    if fullname.endswith('Ext') and mpn[1] == 'mindspore.nn':
                                        new_fullname = 'mindspore.mint.nn.' + fullname[:-3].split('.')[-1]
                                        generate_interface_list.append(
                                            '.. autoclass:: ' + new_fullname + f'&&&{i[0]}')
                                    else:
                                        generate_interface_list.append(
                                            '.. autoclass:: ' + fullname + f'&&&{i[0]}')
                                break
                        else:
                            if k.endswith('.'):
                                generate_interface_list.append(
                                    '.. autofunction:: ' + 'mindspore' + '.' + k.replace('.', '') + f'&&&{i[0]}')
                            elif '.' in k:
                                generate_interface_list.append(
                                    '.. automethod:: ' + 'mindspore' + '.' + k + f'&&&{i[0]}')
                            else:
                                generate_interface_list.append(
                                    '.. autoclass:: ' + 'mindspore' + '.' + k + f'&&&{i[0]}')

                        # break

    return list(set(generate_interface_list))

def supplement_pr_file_cn(pr_cn, repo_path, samedfn_rst, pr_need, base_raw_url, raw_rst_list):
    """
    get same definition apiname from cn files.
    """
    samedfn_cn = []
    for fp in pr_cn:
        if fp == 'need_auto':
            continue
        samedfn_rst_list = []
        filename = fp.split('/')[-1]
        api_name = '.'.join(filename.split('.')[:-1]).replace('.func_', '.')
        api_name_obj = ''
        if 'mindspore.nn.' in api_name or 'mindspore.ops.' in api_name or 'mindspore.mint.' in api_name:
            api_name_obj = get_api(api_name)
        if api_name_obj:
            samedef_list = get_all_samedefinition(api_name_obj, api_name)
            if samedef_list:
                samedfn_rst_list.append(api_name)
            for name in samedef_list:
                mod_name = name.split('.')[1]
                end_name = name.split('.')[-1]
                samedfn_fpath = glob.glob(f'{repo_path}/docs/api/api_python/{mod_name}/*{end_name}.rst')
                ori_p = f'{repo_path}/docs/api/api_python/{mod_name}/{name}.rst'

                for j in samedfn_fpath:
                    rel_filename = j.split(repo_path)[1][1:]
                    if j == ori_p:
                        samedfn_cn.append(rel_filename)
                        raw_rst_list[rel_filename] = f'{base_raw_url}/{rel_filename}'
                        break
                    elif j == ori_p.replace('.func_', '.'):
                        samedfn_cn.append(rel_filename)
                        raw_rst_list[rel_filename] = f'{base_raw_url}/{rel_filename}'
                        break
                else:
                    if 'mindspore.mint.' in name and not re.findall(r'\.nn\.(?!functional).*', name):
                        pr_need.append(name)
                samedfn_rst_list.append(name)
            if samedef_list:
                samedfn_rst.append(sorted(samedfn_rst_list))

    return samedfn_cn

def make_index_rst(target_path, language_f):
    """
    generate index.rst.
    """
    if language_f == 'en':
        title_content = 'PR Document Preview Directory'
    else:
        title_content = 'PR 文档预览目录'

    content = title_content + "\n" + '=' * \
        len(title_content) + '\n\n' + \
        '.. toctree::\n    :glob:\n    :maxdepth: 1\n\n'
    dir_set = set()
    # pylint: disable=W0612
    for rt, dirs, files in os.walk(os.path.join(target_path, 'api_python')):
        for file in files:
            if file.endswith('.rst') and rt.split('api_python')[-1] not in dir_set:
                if not rt.split('api_python')[-1]:
                    content += f"    api_python/{file}\n"
                elif not os.path.basename(rt).startswith('_'):
                    content += f"    api_python{rt.split('api_python')[-1]}/*\n"
                else:
                    continue
                dir_set.add(rt.split('api_python')[-1])

    content += "    api_python/mint/*\n"
    with open(os.path.join(target_path, 'index.rst'), 'w+', encoding='utf-8') as f:
        f.write(content)

def generate_samedfn_rst(samedfn_list):
    """
    Display the same defined interface.
    """
    rst_content = '同定义接口\n================\n\n'
    rst_content += '.. list-table::\n   :widths: 30 30 40\n\n'
    for i in samedfn_list:
        for j in i:
            if j == i[0]:
                rst_content += f'   * - {j}\n'
            else:
                rst_content += f'     - {j}\n'
        if len(i) == 2:
            rst_content += '     -\n'
    return rst_content

def handle_config(pf_cn, pf_py, pf_yaml, pf_sum, target_path, repo_p, pr_need):
    """
    modify config content.
    """
    # 调整config配置
    if pf_cn:
        with open(os.path.join(target_path, 'source_zh_cn/conf.py'), 'r+', encoding='utf-8') as h:
            conf_content = h.read()
            conf_content = conf_content.replace('\ncopy_source(', '\n# copy_source(')
            conf_content = conf_content.replace('primitive_list = ops_interface_name()',
                                                'primitive_list = ops_interface_name()\nprimitive_list = []')
            conf_content = conf_content.replace('\nprimitive_sum = ', f'\nprimitive_sum = []\n# primitive_sum = ')
            conf_content = conf_content.replace('\nmint_sum = ', f'\nmint_sum = {pr_need}\n# mint_sum = ')
            conf_content = conf_content.replace('os.getenv("MS_PATH")', f'"{repo_p}"')
            conf_content = conf_content.replace('import search_code', '# import search_code')
            conf_content = conf_content.replace('import nbsphinx_mod', '# import nbsphinx_mod')
            conf_content = re.sub(r'(generate_ops_mint_rst\(.*)\)', rf'\1, pr_need={pr_need})', conf_content)
            h.seek(0)
            h.truncate()
            h.write(conf_content)

        # 删除api之外的文件夹，加快构建速度
        for i in os.listdir(os.path.join(target_path, 'source_zh_cn')):
            if os.path.isdir(os.path.join(target_path, 'source_zh_cn', i)):
                if not i.startswith('_') and 'api_python' not in i:
                    shutil.rmtree(os.path.join(target_path, 'source_zh_cn', i))

        # 改写index.rst，使生成文档目录
        source_path = os.path.join(target_path, 'source_zh_cn')
        make_index_rst(source_path, 'cn')
    if pf_py or pf_yaml or pf_sum:
        with open(os.path.join(target_path, 'source_en', 'conf.py'), 'r+', encoding='utf-8') as h:
            conf_content = h.read()
            conf_content = conf_content.replace('\ncopy_image(', '\n# copy_image(').replace(
                '\ncopy_source(', '\n# copy_source(')
            conf_content = conf_content.replace('os.getenv("MS_PATH")', f'"{repo_p}"')
            conf_content = conf_content.replace('import search_code', '# import search_code')
            conf_content = conf_content.replace('import nbsphinx_mod', '# import nbsphinx_mod')
            h.seek(0)
            h.truncate()
            h.write(conf_content)
        for i in os.listdir(os.path.join(target_path, 'source_en')):
            if os.path.isdir(os.path.join(target_path, 'source_en', i)) and not i.startswith('_') and i != 'api_python':
                shutil.rmtree(os.path.join(target_path, 'source_en', i))
        # 改写index.rst，使生成文档目录
        source_path = os.path.join(target_path, 'source_en')
        make_index_rst(source_path, 'en')

def generate_version_json(rp_n, branch, js_data, target_path):
    """
    基于base_version.json文件给每个组件生成对应的version.json文件。
    """
    for d in range(len(js_data)):
        if js_data[d]['repo_name'] == rp_n:
            write_content = copy.deepcopy(js_data[d])
            if not write_content['version']:
                write_content['version'] = branch
            write_content.pop("repo_name", None)
            if js_data[d]['repo_name'] != 'mindspore':
                filename = js_data[d]['repo_name']
            else:
                filename = "docs"
            if not branch != "master" and "submenu" in write_content.keys():
                for url in write_content["submenu"]["zh"]:
                    url["url"] = url["url"].replace('/master/', f'/{branch}/')
                for url in write_content["submenu"]["en"]:
                    url["url"] = url["url"].replace('/master/', f'/{branch}/')
            with open(os.path.join(target_path, f"{filename}_version.json"), 'w+', encoding='utf-8') as f:
                json.dump(write_content, f, indent=4)
            break

# mindspore仓API文档生成
def api_generate_prepare(pf_url, pf_diff, rp_dir_docs, rp_dir, clone_branch):
    """
    Preparation before generating API documentation.
    """

    white_list = ['ops/mindspore.ops.comm_note.rst']
    split_dict = {'mindspore_cn': "docs/api/api_python/",
                  'mindspore_en': "docs/api/api_python_en/",
                  'mindspore_py': "mindspore/python/mindspore/",
                  'mindspore_yaml': "mindspore/ops/op_def/yaml/",
                  'mindspore_Tensor_yaml': "mindspore/ops/api_def/method_doc/",
                  'mindspore_function_yaml': "mindspore/ops/api_def/function_doc/"}

    wb_data = requests.get(pf_url)  # 引入requests库来请求数据
    result = wb_data.json()  # 将请求的数据转换为json格式

    # 获取pr文件的diff
    diff = requests.get(pf_diff).text

    cn_flag = 0
    en_flag = 0
    pr_file_py = []
    pr_file_cn = []
    # pr_file_en = []
    pr_file_yaml = []
    auto_need = []
    all_raw_rst = dict()

    generate_pr_list_en_sum = []

    sha_num = result[0]['sha']
    base_raw = f'https://gitee.com/mindspore/mindspore/raw/{sha_num}'

    # pr文件处理
    # pylint: disable=R1702
    for i in range(len(result)):
        filename = result[i]['filename']
        raw_url = result[i]['raw_url']

        # 获取修改文件的内容
        try:
            diff_file = re.findall(f'diff --git .*?{filename}((?:.|\n|)+?)diff --git', diff)[0]
        # pylint: disable=W0702
        except:
            diff_file = re.findall(f'diff --git .*?{filename}((?:.|\n|)+)', diff)[0]
        # 删除一整个文件的跳过
        if '+++ /dev/null' in diff_file:
            continue
        # 记录yaml文件
        # pylint: disable=R1702
        if filename.endswith('.yaml'):
            if split_dict['mindspore_yaml'] in filename or split_dict['mindspore_Tensor_yaml'] in filename:
                pr_file_yaml.append(filename)
            elif split_dict['mindspore_function_yaml'] in filename:
                pr_file_yaml.append(filename)

        # 记录中文API相关文件
        elif split_dict['mindspore_cn'] in filename:
            if filename.endswith('.md') or filename.endswith('.ipynb') or filename.endswith('.rst'):
                file_data = str(requests.get(raw_url).content, 'utf-8')
                # 通过汇总列表新增的接口（未完待续）
                part_atsm = re.findall(r'\.\..*?autosummary::\n\s+?:toctree: (.*)\n(?:.|\n|)+?\n\n((?:.|\n|)+?)\n\n',
                                       file_data+'\n\n')
                if '--- /dev/null' not in diff_file and filename.endswith('.rst'):
                    if os.path.dirname(filename) + '/' == split_dict['mindspore_cn'] and 'autosummary::' in file_data:
                        modify_api = re.findall(r'\+[ ]+?(mindspore\.[\w\.]+?)\n', diff_file)
                        for api_name in modify_api:
                            for sum_p, api_list in part_atsm:
                                if api_name in api_list:
                                    path1 = os.path.join(rp_dir, split_dict['mindspore_cn'], sum_p, api_name+'.rst')
                                    path2 = os.path.join(
                                        rp_dir, split_dict['mindspore_cn'], sum_p,
                                        '.'.join(api_name.split('.')[:-1])+'.func_'+api_name.split('.')[-1]+'.rst')
                                    path3 = os.path.join(
                                        rp_dir, split_dict['mindspore_cn'], sum_p,
                                        '.'.join(api_name.split('.')[:-1])+'.method_'+api_name.split('.')[-1]+'.rst')
                                    if os.path.exists(path1):
                                        file_rel = path1.split(rp_dir)[-1][1:]
                                        pr_file_cn.append(file_rel)
                                        all_raw_rst[file_rel] = f'{base_raw}/{file_rel}'
                                    elif os.path.exists(path2):
                                        file_rel = path2.split(rp_dir)[-1][1:]
                                        pr_file_cn.append(file_rel)
                                        all_raw_rst[file_rel] = f'{base_raw}/{file_rel}'
                                    elif os.path.exists(path3):
                                        file_rel = path3.split(rp_dir)[-1][1:]
                                        pr_file_cn.append(file_rel)
                                        all_raw_rst[file_rel] = f'{base_raw}/{file_rel}'
                                    elif re.findall(r'mindspore\.mint\.(?!nn).*', api_name):
                                        pr_file_cn.append('need_auto')
                                        auto_need.append(api_name)
                                    break
                        continue
                if os.path.exists(os.path.join(rp_dir, filename)) and filename not in white_list:
                    pr_file_cn.append(filename)
                    all_raw_rst[filename] = raw_url
        elif split_dict['mindspore_en'] in filename:
            if not filename.split(split_dict['mindspore_en'])[-1].startswith('mindspore.'):
                continue
            if filename.endswith('.md') or filename.endswith('.ipynb') or filename.endswith('.rst'):
                file_data = str(requests.get(raw_url).content, 'utf-8')
                part_atsm = re.findall(r'\.\..*?autosummary::\n\s+?:toctree: (.*)\n(?:.|\n|)+?\n\n((?:.|\n|)+?)\n\n',
                                       file_data+'\n\n')
                if '--- /dev/null' not in diff_file and filename.endswith('.rst'):
                    if os.path.dirname(filename) + '/' == split_dict['mindspore_en'] and 'autosummary::' in file_data:
                        modify_api = re.findall(r'\+[ ]+?(mindspore\.[\w\.]+?)\n', diff_file)
                        for api_name in modify_api:
                            for sum_p, api_list in part_atsm:
                                if api_name in api_list:
                                    if filename.lower() != filename:
                                        generate_pr_list_en_sum.append(f'.. autoclass:: {api_name}&&&{filename}')
                                    elif '.Tensor.' in api_name:
                                        generate_pr_list_en_sum.append(f'.. automethod:: {api_name}&&&{filename}')
                                    else:
                                        generate_pr_list_en_sum.append(f'.. autofunction:: {api_name}&&&{filename}')
                                    break

        # 记录英文API相关文件
        elif filename.endswith('.py') and split_dict['mindspore_py'] in filename:
            diff_lines = []
            # 新增一整个py文件
            if '--- /dev/null' in diff_file:
                pr_file_py.append(
                    [filename.split(split_dict['mindspore_py'])[-1], diff_lines])
                continue

            diff_arr_num = re.findall(r'@@ .*? \+([0-9]+?),([0-9]+)', diff_file)
            diff_doc = []
            if len(diff_arr_num) == 1:
                diff_doc.append(
                    re.findall(rf'@@ .*? \+{diff_arr_num[0][0]},{diff_arr_num[0][1]} @@((?:.|\n|)+)', diff_file)[0])
            else:
                for k in range(len(diff_arr_num)):
                    dv = diff_arr_num[k]
                    if k+1 == len(diff_arr_num):
                        diff_doc.append(re.findall(
                            rf'@@ .*? \+{diff_arr_num[k][0]},{diff_arr_num[k][1]} @@((?:.|\n|)+)', diff_file)[0])
                    else:
                        dv1 = diff_arr_num[k+1]
                        if re.findall(rf'@@ .*? \+{dv[0]},{dv[1]} @@((?:.|\n|)+?)@@ .*? \+{dv1[0]},{dv1[1]}',
                                      diff_file):
                            diff_doc.append(
                                re.findall(rf'@@ .*? \+{dv[0]},{dv[1]} @@((?:.|\n|)+?)@@ .*? \+{dv1[0]},{dv1[1]}',
                                           diff_file)[0])
                    # else:
                    #     diff_doc.append(re.findall(f'@@ .*? \+{diff_arr_num[k][0]},{diff_arr_num[k][1]} @@ ((?:.|\n|)+)', diff_file)[0])
                    # diff_doc = re.findall('@.*?@@ ((?:.|\n|)+?)@', diff_file)
                    # diff_doc.append(diff_file.split('@@')[-1])

            print(diff_arr_num)

            for j in range(len(diff_arr_num)):
                diff_doc1 = diff_doc[j].replace('\\n', '//n')
                diff_arr1 = min(diff_doc1.find('\n-'), diff_doc1.find('\n+'))
                diff_arr2 = max(diff_doc1.rfind('\n-'), diff_doc1.rfind('\n+'))
                if diff_arr1 == -1:
                    diff_arr1 = max(diff_doc1.find('\n-'),
                                    diff_doc1.find('\n+'))
                addline_num = diff_doc1[:diff_arr1].count('\n')
                delline_num = diff_doc1[diff_arr2:].count('\n')

                if int(diff_arr_num[j][0]) + addline_num > int(diff_arr_num[j][0]) + \
                    int(diff_arr_num[j][1]) - delline_num + 1:
                    diff_lines.append([int(diff_arr_num[j][0]) + int(diff_arr_num[j][1]) -
                                       delline_num + 1, int(diff_arr_num[j][0]) + addline_num])
                else:
                    diff_lines.append([int(diff_arr_num[j][0]) + addline_num, int(
                        diff_arr_num[j][0]) + int(diff_arr_num[j][1]) - delline_num + 1])
            print(diff_lines)

            pr_file_py.append(
                [filename.split(split_dict['mindspore_py'])[-1], diff_lines])

    # docs仓内mindspore工程仓位置
    generate_path = rp_dir_docs + f"/docs/{re.findall('([^/]+?)/pulls/', file_url)[0]}"

    # 没有需要生成的API时直接退出
    if not pr_file_yaml and not pr_file_py and not pr_file_cn and not generate_pr_list_en_sum:
        print('未检测到修改API相关内容，无生成！')
        return generate_path, 0, 0

    all_samedfn_set_en = set()
    all_samedfn_rslist = []

    # 找出中文接口同定义
    samedfn_cn_list = []
    if pr_file_cn:
        samedfn_cn_list = supplement_pr_file_cn(
            pr_file_cn, rp_dir, all_samedfn_rslist, auto_need, base_raw, all_raw_rst)
        pr_file_cn += samedfn_cn_list

    # 提取出修改的英文接口名

    # 为英文生成中文同定义的相关接口
    generate_pr_list_en_samedfn_auto = []
    generate_apien_samedfn_list = []
    for rel_p in samedfn_cn_list:
        samedfn_filename = rel_p.split('/')[-1].replace('.func_', '.')
        samedfn_name = '.'.join(samedfn_filename.split('.')[:-1])
        if samedfn_name.lower() != samedfn_name:
            generate_pr_list_en_samedfn_auto.append(
                f'.. autoclass:: {samedfn_name}&&&samedfn_from_cn')
        else:
            generate_pr_list_en_samedfn_auto.append(
                f'.. autofunction:: {samedfn_name}&&&samedfn_from_cn')
    generate_apien_samedfn_list = get_rst_en(generate_pr_list_en_samedfn_auto, all_samedfn_set_en, all_samedfn_rslist)

    # yaml
    generate_pr_list_en_yaml_auto = []
    generate_apien_yaml_list = []
    if pr_file_yaml:
        generate_pr_list_en_yaml_auto = yaml_file_handle(
            pr_file_yaml, rp_dir, split_dict)
        print(f'从yaml中提取到api的如下：')
        for print_api in generate_pr_list_en_yaml_auto:
            print(print_api)
        generate_apien_yaml_list = get_rst_en(generate_pr_list_en_yaml_auto, all_samedfn_set_en, all_samedfn_rslist)

    # py
    generate_pr_list_en_auto = []
    generate_apien_list = []
    if pr_file_py:
        generate_pr_list_en_auto = en_file_handle(pr_file_py, rp_dir, split_dict)
        print(f'从py文件中提取到的api如下：')
        for print_api in generate_pr_list_en_auto:
            print(print_api)
        generate_apien_list = get_rst_en(generate_pr_list_en_auto, all_samedfn_set_en, all_samedfn_rslist)

    # autosummary
    generate_apien_sum_list = []
    if generate_pr_list_en_sum:
        print(f'从汇总页中提取到的api如下：')
        for print_api in generate_pr_list_en_sum:
            print(print_api)
        generate_apien_sum_list = get_rst_en(generate_pr_list_en_sum, all_samedfn_set_en, all_samedfn_rslist)

    # 清理 api_python 文件夹
    if os.path.exists(os.path.join(generate_path, 'source_en', 'api_python')):
        shutil.rmtree(os.path.join(generate_path, 'source_en', 'api_python'))
    if os.path.exists(os.path.join(generate_path, 'source_zh_cn', 'api_python')):
        shutil.rmtree(os.path.join(generate_path, 'source_zh_cn', 'api_python'))

    all_samedfn_rsset = [list(t) for t in set(tuple(sublist) for sublist in all_samedfn_rslist)]

    print(f'相关同定义接口如下：')
    for print_api in all_samedfn_rsset:
        print(print_api)
    samedfn_content = generate_samedfn_rst(all_samedfn_rsset)

    # 英文文档汇总写入
    all_en_rst = generate_apien_samedfn_list + generate_apien_yaml_list + generate_apien_list + generate_apien_sum_list
    if pr_file_py or pr_file_yaml or generate_apien_sum_list:
        en_set = set()
        for i in all_en_rst:
            if i[0] in en_set:
                continue
            en_set.add(i[0])
            if i[2]:
                if not os.path.exists(os.path.join(generate_path, 'source_en', 'api_python', i[1])):
                    os.makedirs(os.path.join(generate_path, 'source_en', 'api_python', i[1]))
                with open(os.path.join(generate_path, 'source_en', 'api_python', i[1], i[0] + '.rst'),
                          'w+', encoding='utf-8') as f:
                    f.write(i[2])
        en_flag = 1
        copy_source(os.path.join(rp_dir, 'docs/api/api_python_en'), os.path.join(
            generate_path, 'source_en', 'api_python'), 'docs/api/api_python_en')
        copy_image(os.path.join(rp_dir, 'docs/api/api_python'),
                   os.path.join(generate_path, 'source_en', 'api_python'))
        with open(os.path.join(generate_path, 'source_en/api_python/samedfn.rst'), 'w', encoding='utf-8') as f:
            f.write(samedfn_content.replace('同定义接口', 'Same definition interface'))

    # 中文处理
    if all_samedfn_set_en:
        for name in all_samedfn_set_en:
            mod_name = name.split('.')[1]
            end_name = name.split('.')[-1]
            samedfn_fpath = glob.glob(f'{rp_dir}/docs/api/api_python/{mod_name}/*{end_name}.rst')
            ori_p = f'{rp_dir}/docs/api/api_python/{mod_name}/{name}.rst'
            for j in samedfn_fpath:
                rel_filename = j.split(rp_dir)[1][1:]
                if j == ori_p:
                    pr_file_cn.append(rel_filename)
                    all_raw_rst[rel_filename] = f'{base_raw}/{rel_filename}'
                    break
                elif j == ori_p.replace('.func_', '.'):
                    pr_file_cn.append(rel_filename)
                    all_raw_rst[rel_filename] = f'{base_raw}/{rel_filename}'
                    break
            else:
                if 'mindspore.mint.' in name and not re.findall(r'\.nn\.(?!functional).*', name):
                    auto_need.append(name)

    # 自动生成的接口列表
    if auto_need:
        auto_need = list(set(auto_need))
        pr_file_cn.append('need_auto')

    print(f'需要自动生成中文的mint接口如下:')
    for print_api in auto_need:
        print(print_api)

    if pr_file_cn:
        cn_flag = 1
        pr_file_cn = list(set(pr_file_cn))
        print(f'涉及修改的中文api如下：')
        for print_api in pr_file_cn:
            print(print_api)

        copy_file_list = get_all_copy_list(
            pr_file_cn, re.findall('([^/]*?)/pulls/', file_url)[0], clone_branch, rp_dir, all_raw_rst)
        copy_source(os.path.join(rp_dir, 'docs/api/api_python'),
                    os.path.join(generate_path, 'source_zh_cn', 'api_python'),
                    'docs/api/api_python/', fp_list=copy_file_list)
        copy_image(os.path.join(rp_dir, 'docs/api/api_python'),
                   os.path.join(generate_path, 'source_zh_cn', 'api_python'))
        with open(os.path.join(generate_path, 'source_zh_cn/api_python/samedfn.rst'), 'w', encoding='utf-8') as f:
            f.write(samedfn_content)

    handle_config(pr_file_cn, pr_file_py, pr_file_yaml, generate_apien_sum_list, generate_path, rp_dir, auto_need)

    return generate_path, cn_flag, en_flag

def make_html(generate_path, pre_path, cn_flag, en_flag, branch, js_data):
    """
    generate html.
    """

    os.chdir(generate_path)
    if os.path.basename(generate_path) == "mindspore":
        generate_dir = 'docs'
    else:
        generate_dir = os.path.basename(generate_path)

    flush(os.path.join(pre_path, generate_dir))
    # 输出英文
    if en_flag:
        try:
            with open("Makefile", "r+") as f:
                content = f.read()
                content_mod = content.replace("source_zh_cn", "source_en").replace("build_zh_cn", "build_en")
                f.seek(0)
                f.truncate()
                f.write(content_mod)

            cmd_make = ["make", "html"]
            process = subprocess.Popen(cmd_make, stderr=subprocess.PIPE, encoding="utf-8")
            # pylint: disable=W0612
            _, stderr = process.communicate()
            process.wait()
            if process.returncode != 0:
                print(f'stderr: {stderr}')

            TARGET = os.path.join(pre_path, f"{generate_dir}/en/{branch}")
            os.makedirs(os.path.dirname(TARGET), exist_ok=True)
            shutil.copytree("build_en/html", TARGET)
            js_data[0]['English']['link'] = f'{generate_dir}/en/{branch}/index.html'
            js_data[0]['English']['result'] = 'SUCCESS'
            # md_h += f'\n| [English Version]({generate_dir}/en/{branch}/index.html) | SUCCESS |'
        # pylint: disable=W0703
        # pylint: disable=W0702
        except Exception as e:
            print(f"English Version run failed!")
            print(f'Exception: {e}')
            js_data[0]['English']['result'] = 'FAILURE'
            # md_h += '\n| English Version | FAILURE |'

    # 输出中文
    if cn_flag:
        try:
            with open("Makefile", "r+") as f:
                content = f.read()
                content_mod = content.replace("source_en", "source_zh_cn").replace("build_en", "build_zh_cn")
                f.seek(0)
                f.truncate()
                f.write(content_mod)
            cmd_make = ["make", "html"]
            process = subprocess.Popen(cmd_make, stderr=subprocess.PIPE, encoding="utf-8")
            # pylint: disable=W0612
            _, stderr = process.communicate()
            process.wait()
            if process.returncode != 0:
                print(f'stderr: {stderr}')

            TARGET = f"{pre_path}/{generate_dir}/zh-CN/{branch}"
            os.makedirs(os.path.dirname(TARGET), exist_ok=True)
            shutil.copytree("build_zh_cn/html", TARGET)
            js_data[0]['Chinese']['link'] = f'{generate_dir}/zh-CN/{branch}/index.html'
            js_data[0]['Chinese']['result'] = 'SUCCESS'
            # md_h += f'\n| [Chinese Version]({generate_dir}/zh-CN/{branch}/index.html) | SUCCESS |'
        # pylint: disable=W0703
        # pylint: disable=W0702
        except Exception as e:
            print(f"Chinese Version run failed!")
            print(f'Exception: {e}')
            js_data[0]['Chinese']['result'] = 'FAILURE'
            # md_h += '\n| Chinese Version | FAILURE |'
    return js_data, generate_dir

def modify_style_files(pre_path, rp_dn, theme_p, version_p, lge_list):
    """
    copy and modify css, js files.
    """

    output_path = pre_path
    theme_list = []

    if rp_dn == 'docs':
        theme_list.append(rp_dn)
    elif rp_dn == 'tutorials':
        theme_list.append(rp_dn)
    elif rp_dn == 'lite':
        theme_list.append(rp_dn + '/docs')
        theme_list.append(rp_dn + '/api')
    else:
        theme_list.append(rp_dn + '/docs')
    # theme_path = args.theme
    # for f_name in os.listdir(theme_path):
    #     if os.path.isfile(os.path.join(theme_path, f_name)):
    #         if os.path.exists(os.path.join(output_path, f_name)):
    #             os.remove(os.path.join(output_path, f_name))
    #         shutil.copy(os.path.join(theme_path, f_name), os.path.join(output_path, f_name))
    # pylint: disable=W0621
    for lg in lge_list:
        # pylint: disable=W0621
        for out_name in theme_list:
            try:
                static_path_css = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/css/theme.css")[0]
                static_path_js = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/js/theme.js")[0]
                static_path_jquery = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/jquery.js")[0]
                static_path_underscore = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/underscore.js")[0]
                static_path_jquery_ = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/jquery-3.5.1.js")[0]
                static_path_underscore_ = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/underscore-1.13.1.js")
                static_path_underscore_ = static_path_underscore_[0]

                static_path_css_badge = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/css/badge_only.css")[0]
                static_path_js_badge = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/js/badge_only.js")[0]
                static_path_js_html5p = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/js/html5shiv-printshiv.min.js")[0]
                static_path_js_html5 = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/js/html5shiv.min.js")[0]

                static_path_version = glob.glob(f"{output_path}/{out_name}/{lg}/*/_static/js/")[0]
                static_path_version = os.path.join(static_path_version, "version.json")
                if 'lite' in out_name:
                    css_path = f"theme-{out_name.split('/')[0]}/theme.css"
                    js_path = f"theme-{out_name.split('/')[0]}/theme.js"
                elif out_name == 'docs':
                    css_path = "theme-tutorials/theme.css"
                    js_path = "theme-tutorials/theme.js"
                else:
                    css_path = "theme-docs/theme.css"
                    js_path = "theme-docs/theme.js"
                static_path_new_css = os.path.join(theme_p, css_path)
                static_path_new_js = os.path.join(theme_p, js_path)
                static_path_new_jquery = os.path.join(theme_p, "update_js", "jquery.js")
                static_path_new_underscore = os.path.join(theme_p, "update_js", "underscore.js")
                out_name_1 = out_name.split('/')[0]
                static_path_new_version = os.path.join(version_p, f"{out_name_1}_version.json")
                fonts_dir_1 = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/fonts/")
                fonts_dir_2 = glob.glob(
                    f"{output_path}/{out_name}/{lg}/*/_static/css/fonts/")
                if fonts_dir_1 and os.path.exists(fonts_dir_1[0]):
                    shutil.rmtree(fonts_dir_1[0])
                if fonts_dir_2 and os.path.exists(fonts_dir_2[0]):
                    shutil.rmtree(fonts_dir_2[0])
                if os.path.exists(static_path_css):
                    os.remove(static_path_css)
                shutil.copy(static_path_new_css, static_path_css)
                if os.path.exists(static_path_js):
                    os.remove(static_path_js)
                shutil.copy(static_path_new_js, static_path_js)
                if os.path.exists(static_path_version):
                    os.remove(static_path_version)
                shutil.copy(static_path_new_version, static_path_version)
                if os.path.exists(static_path_jquery):
                    os.remove(static_path_jquery)
                shutil.copy(static_path_new_jquery, static_path_jquery)
                if os.path.exists(static_path_underscore):
                    os.remove(static_path_underscore)
                shutil.copy(static_path_new_underscore, static_path_underscore)
                if os.path.exists(static_path_jquery):
                    os.remove(static_path_jquery_)
                if os.path.exists(static_path_underscore_):
                    os.remove(static_path_underscore_)
                if os.path.exists(static_path_css_badge):
                    os.remove(static_path_css_badge)
                if os.path.exists(static_path_js_badge):
                    os.remove(static_path_js_badge)
                if os.path.exists(static_path_js_html5p):
                    os.remove(static_path_js_html5p)
                if os.path.exists(static_path_js_html5):
                    os.remove(static_path_js_html5)

            # pylint: disable=W0702
            # pylint: disable=W0703
            except Exception as e:
                print(f'replace {out_name} dir style files failed!\n{e}')
                continue
    print(f'replace style files end!')

if __name__ == "__main__":

    # 添加命令行参数以供使用
    parser = argparse.ArgumentParser()
    parser.add_argument('--pr_url', type=str, default="")  # pr网址
    parser.add_argument('--whl_path', type=str, default="") # 安装包地址
    args = parser.parse_args()

    pr_url = args.pr_url
    pr_whl_path = args.whl_path

    # 参数处理（准备工作）
    file_url = pr_url.split('.com/')[-1].split('/files')[0]
    apiv5_url = f'https://gitee.com/api/v5/repos/{file_url}'
    diff_url = f'https://gitee.com/{file_url}.diff'
    pr_comments_url = f'{apiv5_url}/comments?page=1&per_page=100&direction=desc'
    pr_files_url = f'{apiv5_url}/files'

    # docs仓和repo仓路径
    present_dir_path = os.path.dirname(os.path.abspath(__file__))
    theme_path = os.path.join(present_dir_path, 'template')
    repo_dir_docs = os.path.join(present_dir_path, '../../../docs')
    repo_name = re.findall('([^/]*?)/pulls/', file_url)[0]
    repo_dir = os.path.join(present_dir_path, f'../../../{repo_name}')

    # 获取pr合入的分支
    res = requests.get(apiv5_url).text
    repo_branch = re.findall('"base":{"label":"(.+?)"', res)[0]

    # 切换本地仓库分支
    docs_branch = update_repo(repo_branch, repo_dir_docs)

    # 安装依赖包
    cmd_uninstall = ["pip", "uninstall", "-y", "mindspore"]
    subprocess.run(cmd_uninstall)
    cmd_install = ["pip", "install", pr_whl_path]
    subprocess.run(cmd_install)

    # 生成version.json
    with open(os.path.join(present_dir_path, "base_version.json"), 'r+', encoding='utf-8') as g:
        data_b = json.load(g)
    target_version = os.path.join(present_dir_path, f"{docs_branch}_version")
    flush(target_version)
    generate_version_json(repo_name, docs_branch, data_b, target_version)

    # mindspore仓pr修改准备
    mk_ht_path, cn_f, en_f = api_generate_prepare(
        pr_files_url, diff_url, repo_dir_docs, repo_dir, repo_branch)

    # 开始构建文档
    # md_head = '| API Documentation | Build Result |'
    with open(os.path.join(present_dir_path, "api_result.json"), 'r+', encoding='utf-8') as g:
        result_data = json.load(g)
    g_lan = []
    if cn_f:
        g_lan.append('zh-CN')
    if en_f:
        g_lan.append('en')

    # 屏蔽sphinx 在python>=3.9时额外依赖引入的版本过高问题
    pythonlib_dir = os.path.dirname(os.path.dirname(sphinx.__file__))
    registry_target = os.path.join(pythonlib_dir, 'sphinx', 'registry.py')
    with open(registry_target, 'r+', encoding='utf-8') as g:
        registry_content = g.read()
        registry_content = re.sub(r'([ ]+?)except VersionRequirementError as err:\n(?:.|\n|)+?from err',
                                  r'\1except VersionRequirementError as err:\n\1    metadata = {}',
                                  registry_content)
        g.seek(0)
        g.truncate()
        g.write(registry_content)

    js_content, generate_dir_name = make_html(mk_ht_path, present_dir_path, cn_f, en_f, docs_branch, result_data)

    # 修改样式文件
    modify_style_files(present_dir_path, generate_dir_name, theme_path, target_version, g_lan)
    os.chdir(present_dir_path)
    cmd_tar = ["tar", "-czvf", f"{generate_dir_name}.tar.gz", f'./{generate_dir_name}']
    subprocess.run(cmd_tar)

    with open(os.path.join(present_dir_path, "api_result.json"), 'w+', encoding='utf-8') as g:
        json.dump(js_content, g, indent=4)
