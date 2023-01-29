"""remove div wrapper for svg element in mindquantum jupyter notebook."""
import os

ext_path = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.abspath(os.path.join(ext_path, os.pardir))
source_en = os.path.join(docs_path, "source_en")
source_zh_cn = os.path.join(docs_path, "source_zh_cn")


def get_all_ipynb(source_path):
    """
    Get all jupyter notebook file in source_path
    """
    out = []
    for root, _, files in os.walk(source_path):
        for file in files:
            if file.endswith('.ipynb'):
                file_path = os.path.join(root, file)
                out.append(file_path)
    return out


def remove_div(file_name):
    """
    Remove the div wrapper around svg output for mindquantum.
    """
    a = "<div class=\\\"nb-html-output output_area\\\">"
    b = "</div>"
    out = []
    with open(file_name, 'r', encoding="utf8") as f:
        for code_str in f.readlines():
            if a in code_str:
                new_code = code_str.replace(a, '')
                c1 = new_code[:-9]
                c2 = new_code[-9:]
                c2 = c2.replace(b, '')
                out.append(c1 + c2)
            else:
                out.append(code_str)
    with open(file_name, 'w', encoding='utf8') as f:
        f.writelines(out)


all_path = []
all_path.extend(get_all_ipynb(source_en))
all_path.extend(get_all_ipynb(source_zh_cn))
for i in all_path:
    remove_div(i)
