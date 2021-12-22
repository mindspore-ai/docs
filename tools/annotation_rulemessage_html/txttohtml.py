'''rulemessage_txt to html'''
import re
import os

def run(txtinput_filepath):
    '''txt to html'''
    txtinput_name = txtinput_filepath.split("/")[-1].split(".")[0]
    res_str = re.compile(r"((\d+)\.\) Line (\d+).*?Rule ID: (.*?)\nMessage: (.*?)\n(Suggestion: (.*?)\n)?(.*)\n)")
    result = '<h2>' + str(txtinput_filepath.split("-")[0]) + '.' + str(txtinput_filepath.split(".")[1]) + '</h2>\n'
    result = result + '''<table border='1' class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th>序号</th>
          <th>行号</th>
          <th>规则ID</th>
          <th>出错的信息</th>
          <th>可替换的词</th>
          <th>原文内容</th>
        </tr>
      </thead>
      <tbody>'''
    with open(txtinput_filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        patterned = res_str.findall(content)
        if patterned:
            for i in patterned:
                num, line, id_, msg, sug, text = "", "", "", "", "", ""
                if "Suggestion" in i[0]:
                    num, line, id_, msg, sug, text = i[1], i[2], i[3], i[4], i[6], i[7]
                else:
                    num, line, id_, msg, text = i[1], i[2], i[3], i[4], i[7]
                result = result + '<tr>\n<th>' + num + \
                '</th>\n<td>' + line + '</td>\n<td>' + id_ + \
                '</td>\n<td>' + msg + '</td>\n<td>' + sug + \
                '</td>\n<td>' + text + '</td>\n</tr>'
    result = result + '''  </tbody>
    </table>'''
    txtoutput_path = "/".join(str(txtinput_filepath).split("/")[:-1])+"/{}.html".format(txtinput_name)
    txtoutput_dir = "/".join(txtoutput_path.split("/")[:-1])
    if not os.path.exists(txtoutput_dir):
        os.makedirs(txtoutput_dir)
    with open(txtoutput_path, "w", encoding='utf-8') as rf:
        rf.writelines(result)
