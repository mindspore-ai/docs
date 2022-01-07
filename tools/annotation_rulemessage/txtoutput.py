'''format rulemessagetxt'''
#(\d+)\.\) Line (\d+).*?Rule ID: (.*?)\nMessage: (.*?)\n(Suggestion: (.*?)\n)?(.*)\n

import re
import os

def run(txtinput_filepath):
    '''format rulemessage'''
    txtinput_name = txtinput_filepath.split("/")[-1].split(".")[0]
    res_str = re.compile(r"((\d+)\.\) Line (\d+).*?Rule ID: (.*?)\nMessage: (.*?)\n(Suggestion: (.*?)\n)?(.*)\n)")
    result = '\n' + str(txtinput_filepath) + '\n'
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
                result = result + 'Num：' + num + '\tLine：' \
                + line + '\tRuleID：' + id_ + '\tMessage：' \
                + msg + '\tSuggestion：' + sug + '\tContext：' \
                + text + '\n'
    txtoutput_path = "formatrule/" \
    +"/".join(str(txtinput_filepath).split("/")[:-1]) \
    + "/{}-outputrule.txt".format(txtinput_name)
    txtoutput_dir = "/".join(txtoutput_path.split("/")[:-1])
    if not os.path.exists(txtoutput_dir):
        os.makedirs(txtoutput_dir)
    with open(txtoutput_path, "w", encoding='utf-8') as rf:
        rf.writelines(result)
