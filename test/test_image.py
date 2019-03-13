#coding=utf-8

#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np

if __name__=="__main__":
    datas = []
    a="E:\\aaaaa\\download_captcha\\2222\\1.png"
    image = Image.open(a)
    data = np.array(image)
    datas.append(data)
    datas = np.array(datas)
    print(len(datas))
    print(datas.shape)
    b="E:\\aaaaa\\download_captcha\\2222\\2.jpg"
    image = Image.open(b)
    data = np.array(image)
    datas = []
    datas.append(data)
    datas = np.array(datas)
    print(len(datas))
    print(datas.shape)