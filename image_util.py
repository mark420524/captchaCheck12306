# coding=utf-8
from PIL import Image,ImageFile
import numpy as np
import os
import time
import requests
import uuid
# broken data stream when reading image file 文件损坏修复
ImageFile.LOAD_TRUNCATED_IMAGES = True
session=requests.Session()
requests.packages.urllib3.disable_warnings()
imageDir = "temp/"
curtImg = "curt1/"
file_type=".png"
curtDir="curt_words/"
def judgeImageBackground(image):
    """
    判断验证码文字区域的词个数
    :param image: Image对象或图像路径
    :return: 1 或 2
    """
    if isinstance(image, str):
        raw_image = Image.open(image)

    # 裁切出验证码文字区域 高28 宽112
    image = raw_image.crop((118, 0, 230, 28))
    image = image.convert("P")
    image_array = np.asarray(image)

    image_array = image_array[24:28]
    #print(image_array)
    if np.mean(image_array) > 200:
        return 1
    else:
        return 2
def splitImageText( image, image_shape, mode=1):
    """
    裁切出验证码文字部分
    :param image: Image对象或图像路径
    :param mode: 图中有几组验证码文字
    :return:
    """
    
    if isinstance(image, str):
        raw_image = Image.open(image)
    # 裁切出验证码文字区域 高28 宽112
    image = raw_image.crop((118, 0, 230, 28))
    #save(image,"temp/","curt1/",image_shape)
    resize_list = []
    if mode == 1:
        # 图中只有一组验证码
        image_array = np.asarray(image)
        image_array = image_array[6:22]
        image_array = np.mean(image_array, axis=2)
        image_array = np.mean(image_array, axis=0)
        image_array = np.reshape(image_array, [-1])

        indices = np.where(image_array < 240)
        resize_list.append((indices[0][0], indices[0][-1]))

    if mode == 2:
        # 图中只有两组验证码
        image_p = image.convert("P")
        image_array = np.asarray(image_p)
        image_array = image_array[6:22]
        image_array = np.mean(image_array, axis=0)
        avg_image = np.reshape(image_array, [-1])
        indices = np.where(avg_image < 190)
        start = indices[0][0] - 1
        end = indices[0][0] - 1
        for i in indices[0]:
            if i == end + 1:
                end = i
            else:
                if end - start > 10:
                    resize_list.append([start + 1, end])
                start = i
                end = i
        if end - start > 10:
            resize_list.append([start + 1, end])
    cutImage = [image.crop((x1, 0, x2, 28)) for x1, x2 in resize_list]
    cutImageName = []
    for x3 in cutImage:
        cutImageName.append(save(x3,curtDir,curtImg ))
    return cutImageName
     
    
def save(image, dir, label, image_shape=(64, 64)):
    """
    保存图片
    :param image:
    :param dir:  保存目录
    :param label: 子文件夹
    :param image_shape:
    :return:
    """
    if not os.path.exists(dir):
        os.mkdir(dir)

    path = os.path.join(dir, label)
    if not os.path.exists(path):
        os.mkdir(path)

    filename = "%s.png" % uuid.uuid4()
    #image = image.resize(image_shape)
    image.save(os.path.join(path, filename))
    return dir+label+filename
def curtDirImage(dir,image_shape=(64, 64)):
    #path=os.path.join(dir, '')
    count = 0
    for root,dirs,files in os.walk(dir, topdown=False):
        for file in files:
            f,e=os.path.splitext(file)
            if e==file_type:
                count+=1
                image_path = os.path.join(root,file)
                #print(image_path)
                flag = judgeImageBackground(image_path)
                splitImageText(image_path,image_shape,flag)
    return count

def read_image(image_dir,image_shape=None,label_path="label.txt"):
    """
     读取图片
     :params image_dir: 图片路径
     :params image_shape: 图片宽高dict 
     :params label_path: 12306图片分类label

    """
    label_object={}
    if os.path.exists(label_path):
        with open(label_path,encoding="utf-8") as files:
            for lines in files:
                # split 默认已空格拆分字符串
                lable_name,id = lines.strip().split()
                label_object[lable_name]=int(id)
    else:
        with open(label_path,"w",encoding="utf-8") as files:
            for id, lable_name in enumerate(os.listdir(image_dir)):
                # split 默认已空格拆分字符串
                files.write("%s %s\n" % (lable_name, id))
                #lable_name,id = lines.strip().split()
                label_object[lable_name]=int(id)
    # 训练的图像数组
    train_images = []
    # 训练的分类数组
    train_labels = [] 
    file_label_dict={}
    for dir_name in os.listdir(image_dir):
        for file_name in os.listdir(os.path.join(image_dir, dir_name)):
            full_image_path = os.path.join(image_dir,dir_name,file_name)
            current_label = label_object[dir_name]
            file_label_dict[full_image_path]=current_label
    # 获取所有图片全路径
    keys = list(file_label_dict.keys())
    #打乱训练图片顺序
    np.random.shuffle(keys)

    for file_path in keys:
        image = Image.open(file_path)

        if image_shape and image.size != image_shape[:2]:
            image = image.resize(image_shape[:2])
        image = np.asarray(image)
        train_images.append(image)
        train_labels.append(file_label_dict[file_path])
    return np.array(train_images),np.array(train_labels)
if __name__ == '__main__':
    pass
    #image_dir="E:\\aaaaa\\download_captcha\\curt_words\\traindata\\安全帽\\5072c75d-1911-459e-adc4-308f71f76f9e.png"
    #flag=judgeImageBackground(image_dir)
     
    #print(read_image(image_dir))
    #imageShape = (64,64)
    #cutImageName = splitImageText(image_dir,imageShape,flag)
    
