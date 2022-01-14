'''python execute languagetool-commandline.jar'''
# pylint: disable=W0106, W0612
import os
import sys
import subprocess

def run(file_path):
    """export result"""
    file_name = file_path.split("/")[-1].split(".")[0]
    out_path = "rulemessage/"+"/".join(str(file_path).split("/")[:-1])+"/{}-rulemessage.txt".format(file_name)
    out_dir = "/".join(out_path.split("/")[:-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    arg0 = sys.argv[2]
    arg1 = '-l en-US'
    arg2 = '-d WHITESPACE_RULE,COMMA_PARENTHESIS_WHITESPACE,SENTENCE_WHITESPACE,\
UPPERCASE_SENTENCE_START,EN_UNPAIRED_BRACKETS,DOUBLE_PUNCTUATION,LC_AFTER_PERIOD[1],\
ARROWS,I_LOWERCASE[1],MORFOLOGIK_RULE_EN_US,UNLIKELY_OPENING_PUNCTUATION[1],PLUS_MINUS,\
ID_CASING,DASH_RULE,ENGLISH_WORD_REPEAT_BEGINNING_RULE,UNIT_SPACE'
    arg3 = file_path
    arg4 = out_path
    command = 'java -jar %s %s %s %s > %s' %(arg0, arg1, arg2, arg3, arg4)
    stdout = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0]
