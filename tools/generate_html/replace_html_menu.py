"""修改文档页面左侧大纲目录"""
import re
import os

# pylint: disable=C0301

# 为有index的目录生成基础大纲目录 --> dict
def replace_html_menu(html_path, hm_ds_path):
    """替换左侧目录内容"""
    rp_dict = dict()
    # pylint: disable=W0621
    # pylint: disable=R1702
    # pylint: disable=W0612
    for root, dirs, files in os.walk(hm_ds_path):
        for file in files:
            if file == 'index.rst' and root != hm_ds_path:
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                with open(os.path.join(root.replace(hm_ds_path, html_path), 'index.html'), 'r', encoding='utf-8') as g:
                    h_content = g.read()
                rp_str = ''
                if len(re.findall('.. toctree::', content)) == 1:
                    toc_docs = re.findall('.. toctree::(?:.|\n|)+?\n\n((?:.|\n|)+?)\n\n', content+'\n\n')[0]
                    toc_list = [i.strip() for i in toc_docs.split('\n') if i.strip() != '']
                    for i in toc_list:
                        title_name = re.findall(
                            f'<li class="toctree-l[0-9]"><a class="reference internal" href="{i}.html">(.*?)</a></li>',
                            h_content)
                        if title_name:
                            rp_str += f'<li class="toctree-l1"><a class="reference internal" href="{i}.html">' +\
                                title_name[0]+'</a></li>\n'
                    # 特殊页面处理
                    if '/migration_guide' in root:
                        rp_dict[os.path.join(html_path, 'note', 'api_mapping')] = rp_str
                    elif '/api_python' in root:
                        rp_dict[os.path.join(html_path, 'note')] = rp_str
                else:
                    toc_docs = re.findall('.. toctree::(?:.|\n|)+?:caption: (.*)\n\n((?:.|\n|)+?)\n\n', content+'\n')
                    for title, toc_doc in toc_docs:
                        toc_list = [i.strip() for i in toc_doc.split('\n') if i.strip() != '']
                        rp_str += '<p class="caption" role="heading"><span class="caption-text">'+\
                            title + '</span></p>\n<ul>\n'
                        for i in toc_list:
                            if '.html' in i:
                                spec_html = re.findall(r'\<(.*?)\>', i)[0]
                                spec_title = i.split(' ')[0]
                                rp_str += f'<li class="toctree-l1"><a class="reference external" href="{spec_html}">'+\
                                    spec_title + '</a></li>\n'
                                continue
                            url_title = re.findall(
                                f'<li class="toctree-l[0-9]"><a class="reference internal" href="(.*?{i}.html)">(.*?)</a></li>',
                                h_content)
                            if url_title:
                                title_name = url_title[0][1]
                                rp_str += f'<li class="toctree-l1"><a class="reference internal" href="{i}.html">'+\
                                    title_name + '</a></li>\n'
                        rp_str += '</ul>\n'

                rp_dict[root.replace(hm_ds_path, html_path)] = rp_str

    # 遍历html页面，替换目录部分
    # pylint: disable=W0621
    for root, dirs, files in os.walk(html_path):
        for file in files:
            if not file.endswith('.html'):
                continue
            if file == 'RELEASE.html':
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    h_content = f.read()
                release_str = '<li class="toctree-l1 current"><a class="current reference internal"'+\
                    ' href="#">Release Notes</a></li>'
                new_content = re.sub(
                    '<ul class="current">(?:.|\n|)+?</ul>\n\n', f'<ul class="current">\n{release_str}</ul>\n\n',
                    h_content)
                with open(os.path.join(root, file), 'w', encoding='utf-8') as f:
                    f.write(new_content)
                continue
            for p in rp_dict:
                if p in root:
                    p_key = p
                    if 'note/api_mapping' in root and p_key+'/api_mapping' in rp_dict:
                        p_key = p+'/api_mapping'
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        h_content = f.read()
                    rp_str = rp_dict[p_key]
                    # 将每个页面的链接修正成正确的相对路径
                    for href in re.findall('href="(.*?)"', rp_str):
                        abs_p1 = os.path.join(p_key, href)
                        if 'note/api_mapping' in p_key:
                            abs_p1 = os.path.join(html_path, 'migration_guide', href)
                        elif '/note' in p_key and 'api_mapping' not in root:
                            abs_p1 = os.path.join(html_path, 'api_python', href)
                        rel_p = os.path.relpath(abs_p1, root)
                        rel_p = rel_p.replace('\\', '/')
                        rp_str = re.sub(f'href="{href}"', f'href="{rel_p}"', rp_str)

                    current_herf = re.findall(
                        '<li class="toctree-l[^1]( current"><a class=".*?reference internal" href="(.*?)">)',
                        h_content)
                    if 'api_python' in root and file.startswith('mindspore.') and current_herf:
                        rp_str = re.sub(
                            f'<li class="toctree-l1"><a class="reference internal" href="{current_herf[0][1]}">',
                            f'<li class="toctree-l1{current_herf[0][0]}', rp_str)

                        rp_str = rp_str.replace('toctree-l3', 'toctree-l2')
                        rp_str = rp_str.replace('toctree-l4', 'toctree-l3')

                    else:
                        # 拥有子目录的文件的特殊处理
                        file_title = re.findall(f'<li class="toctree-l1"><a class="reference internal" href="{file}">(.*?)</a>', rp_str)
                        if file_title and file != 'index.html':
                            toctree_lev = re.findall(f'<li class="toctree-l([0-9]) current"><a class="current reference internal" href="#">{file_title[0]}</a><ul>', h_content)
                            extra_re = []
                            if toctree_lev:
                                extra_re = re.findall(f'<li class="toctree-l{toctree_lev[0]} current"><a class="current reference internal" href="#">{file_title[0]}</a>(<ul>(?:.|\n|)+?</ul>)\n</li>\n<li class="toctree-l{toctree_lev[0]}', h_content)
                                if not extra_re:
                                    extra_re = re.findall(f'<li class="toctree-l{toctree_lev[0]} current"><a class="current reference internal" href="#">{file_title[0]}</a>(<ul>(?:.|\n|)+?</ul>)\n</li>\n</ul>\n</li>\n<li class="toctree-l1', h_content)
                                if not extra_re:
                                    extra_re = re.findall(f'<li class="toctree-l{toctree_lev[0]} current"><a class="current reference internal" href="#">{file_title[0]}</a>(<ul>(?:.|\n|)+?</ul>)\n</li>\n</ul>\n</li>\n</ul>\n\n', h_content)
                            if extra_re:
                                extra_ul = '<ul>\n' + '\n'.join(re.findall('<li class="toctree-l[0-9]">.*?href="[^#].*?</li>', extra_re[0])) + '\n</ul>'

                                extra_ul = re.sub('toctree-l[0-9]', 'toctree-l2', extra_ul)
                                rp_str = rp_str.replace(f'<li class="toctree-l1"><a class="reference internal" href="{file}">{file_title[0]}</a></li>',
                                                        f'<li class="toctree-l1 current"><a class="current reference internal" href="#">{file_title[0]}</a>{extra_ul}</li>')
                        # 子目录文件的特殊处理
                        elif file != 'index.html':
                            toctree_lev = re.findall(
                                '<li class="toctree-l([^1]) current"><a class="reference internal" href=".*?">.*?</a><ul class="current">',
                                h_content)
                            extra_re = []
                            if toctree_lev:
                                extra_re = re.findall(
                                    rf'<li class="toctree-l{toctree_lev[0]} current">(<a class="reference internal" href=".*?">.*?</a>)<ul class="current">((?:.|\n|)+?)</li>\n<li class="toctree-l{toctree_lev[0]}">',
                                    h_content)
                                if not extra_re:
                                    extra_re = re.findall(
                                        rf'<li class="toctree-l{toctree_lev[0]} current">(<a class="reference internal" href=".*?">.*?</a>)<ul class="current">((?:.|\n|)+?)</li>\n</ul>\n</li>\n<li class="toctree-l1',
                                        h_content)
                                if not extra_re:
                                    extra_re = re.findall(
                                        rf'<li class="toctree-l{toctree_lev[0]} current">(<a class="reference internal" href=".*?">.*?</a>)<ul class="current">((?:.|\n|)+?)</li>\n</ul>\n</li>\n</ul>\n\n">',
                                        h_content)
                            if extra_re:
                                extra_ul = extra_re[0][1]
                                toctree_lev_pre = re.findall(
                                    '<li class="toctree-l([^1]) current"><a class="current reference internal" href="#">',
                                    h_content)
                                if toctree_lev_pre:
                                    extra_pre = re.findall(
                                        rf'(<li class="toctree-l{toctree_lev_pre[0]}.*?href="([^#]+?|#)".*?</a>)',
                                        h_content)
                                    extra_ul = '<ul class="current">\n'
                                    for li_d in extra_pre:
                                        extra_ul += li_d[0].replace(f'toctree-l{toctree_lev_pre[0]}', 'toctree-l2') +\
                                            '</li>\n'

                                    extra_ul += '</ul>\n'

                                else:
                                    # no #
                                    for html_li in re.findall('<li class="toctree-l[0-9].*?">.*?href=".*?">.*?</a></li>', extra_re[0][1]):
                                        if re.findall('href="#"', html_li) or re.findall('href="[^#]+?"', html_li):
                                            continue
                                        else:
                                            extra_ul = extra_ul.replace(html_li+'\n', '')

                                    extra_ul = '<ul class="current">' + extra_ul

                                    extra_ul = extra_ul.replace('toctree-l3', 'toctree-l2')
                                    extra_ul = extra_ul.replace('toctree-l4', 'toctree-l3')
                                rp_str = re.sub(f'(<li class="toctree-l[0-9])">{extra_re[0][0]}', rf'\1 current">{extra_re[0][0]}{extra_ul}', rp_str)

                    rp_str = re.sub(r'<ul>[\n ]+?</ul>', '', rp_str)
                    # 选中当前页面的显示
                    rp_str = rp_str.replace(f'<li class="toctree-l1"><a class="reference internal" href="{file}">',
                                            '<li class="toctree-l1 current"><a class="current reference internal" href="#">')
                    new_content = re.sub(r'<ul class="current">(?:.|\n|)+?</ul>\n\n', f'<ul class="current">\n{rp_str}</ul>\n\n', h_content)

                    if new_content == h_content:
                        new_content = re.sub(
                            r'(data-spy="affix" role="navigation" aria-label="Navigation menu">[\n ]+?)<ul>(?:.|\n|)+?</ul>\n\n',
                            rf'\1<ul>\n{rp_str}</ul>\n\n', h_content)

                    if '<p class="caption"' in rp_str:
                        new_content = re.sub(
                            r'(data-spy="affix" role="navigation" aria-label="Navigation menu">[\n ]+?)<ul[^\^]+?>(?:.|\n|)+?</ul>\n\n',
                            rf'\1\n{rp_str}\n\n', new_content)

                    if new_content != h_content:
                        if file == 'index.html':
                            new_content = new_content.replace(
                                'data-spy="affix" role="navigation" aria-label="Navigation menu">',
                                'data-spy="affix" role="navigation" aria-label="Navigation menu" docs-caption="">')
                        with open(os.path.join(root, file), 'w', encoding='utf-8') as f:
                            f.write(new_content)
                    else:
                        print(os.path.join(root, file))
                    break
