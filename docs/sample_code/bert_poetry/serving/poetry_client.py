import time
import json
import requests
import argparse
import re

headers = {"content-type": "application/json"}
url = 'http://10.155.170.71:8080/'

while True:
    print("\n**********************************")
    type = input("选择模式：0-随机生成，1：续写，2：藏头诗\n")
    try:
        type = int(type)
    except:
        continue
    if type not in [0, 1, 2]:
        continue
    if type == 1:
        s = input("输入首句诗\n")
    elif type == 2:
        s = input("输入藏头诗\n")
    else:
        s = ''
        
    data = json.dumps({'string': s, 'type': type})
    start_time = time.time()
    json_response = requests.post(url, data=data, headers=headers)
    end_to_end_delay = (time.time()-start_time)*1000
    predictions = json_response.text
    a = re.findall(r'[\u4e00-\u9fa5]*[\uff0c\u3002]', predictions)
    print("\n")
    for poem in a:
        print(poem)
    print("\ncost time: {:.1f} ms".format(end_to_end_delay))

