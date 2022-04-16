"""Modify nbsphinx."""
import os
import nbsphinx as nbs


short_title_define = '''\
import re
def short_title(titlename):
    zhPattern = re.findall(r'[\u4e00-\u9fa5]',titlename)
    if not zhPattern and "." in titlename:
        if titlename==titlename.lower():
            pass
        else:
            return titlename
    titlename = titlename.lower()
    cn_symbol = "。；，“”（）、？《》"
    en_symbol = ".,=+*()<>[]{}|/&"
    spec_symbol = "_:："
    for i in cn_symbol:
        titlename = titlename.replace(i, '')
    for j in en_symbol:
        titlename = titlename.replace(j, '')
    for k in spec_symbol:
        titlename = titlename.replace(k, ' ')
    titlename = titlename.replace('  ', ' ').replace(' ', '-')

    return titlename
'''

# Remove extra outputs for nbsphinx extension.
nbsphinx_source = r"    app.connect('html-collect-pages', html_collect_pages)\n"
before = "link_id = title.replace(' ', '-')"
after = "link_id = short_title(title)"
mod_path = os.path.abspath(nbs.__file__)
with open(mod_path, "r+", encoding="utf8") as f:
    contents = f.read()
    contents.replace(nbsphinx_source, '')
    contents = contents.replace(before, after)
    #pylint: disable=exec-used
    exec(short_title_define, nbs.__dict__)
    exec(contents, nbs.__dict__)
