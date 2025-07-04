"""Check for unused images and delete them."""
import subprocess
import os

def get_images_dierctory(check_path):
    '''
    get all images directory.
    '''
    cmd = ["find", check_path, "-type", "d", "-name", "images"]
    res = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    dir_list = res.stdout.read().decode('utf-8').split('\n')
    del dir_list[-1]
    return dir_list

def get_all_pic(dir_list):
    '''
    get all the images in the images directory.
    '''
    for dir_p in dir_list:
        res = subprocess.Popen(['ls', dir_p], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        pic_list = res.stdout.read().decode('utf-8').split('\n')
        del pic_list[-1]
        for i in pic_list:
            pic_all.add(i)

def get_use_pic(check_path):
    '''
    get all the useful pictures.
    '''
    cmd1 = ['find', check_path, '-type', 'f', '-name', '"*.md"']
    cmd2 = ['find', check_path, '-type', 'f', '-name', '"*.ipynb"']
    cmd3 = [cmd1, cmd2]
    for i in cmd3:
        res = subprocess.Popen(i, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        file_list = res.stdout.read().decode('utf-8').split('\n')
        del file_list[-1]
        for j in file_list:
            j = j.split('/', 1)[1].replace('/', ':/', 1)
            with open(j, 'r', encoding='utf-8') as f:
                data = f.read()
                for k in pic_all:
                    if k in data:
                        use_pic.add(k)

def get_use_eddx():
    '''
    get all the useful eddx files.
    '''
    for i in filter_pic:
        if i.endswith('eddx'):
            if i.split('.')[0] in ' '.join(use_pic):
                use_eddx.add(i)

def get_useless_pic_path(check_path):
    '''
    get the absolute path of all useless pictures.
    '''
    for i in useless_pic:
        cmd = ['find', check_path, '-type', 'f', '-name', i]
        res = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        data = res.stdout.read().decode('utf-8').split('\n')
        del data[-1]
        for j in data:
            path.append(j)

def del_useless_pic():
    '''
    delete all useless pictures.
    '''
    for i in path:
        os.system('rm -rf %s' %i)


if __name__ == '__main__':
    check_path_input = input('请输入您要检测的绝对路径：').strip()
    pic_all = set()
    use_pic = set()
    use_eddx = set()
    path = []
    dir_list_g = get_images_dierctory(check_path_input)
    get_all_pic(dir_list_g)
    get_use_pic(check_path_input)
    filter_pic = pic_all.difference(use_pic)
    get_use_eddx()
    useless_pic = filter_pic.difference(use_eddx)
    get_useless_pic_path(check_path_input)
    print('没有用的照片：', path)
    del_useless_pic()
    print('删除成功')
