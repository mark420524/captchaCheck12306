# coding=utf-8
from PIL import Image
import numpy as np
import os
import time
import requests
import uuid

session=requests.Session()
requests.packages.urllib3.disable_warnings()
imageDir = "temp/"
curtImg = "curt1/"

def downloadCaptcha(retry=False):
    """
    下载12306验证码图片
    :param retry:
    :return:
    """
    url = "https://kyfw.12306.cn/passport/captcha/captcha-image"
    r = session.get(url, verify=False)
    if retry:
        urlBak = "https://kyfw.12306.cn/passport/captcha/captcha-check?answer=129%2C122%2C175%2C132&login_site=E&rand=sjrand"
        time.sleep(3)
        session.get(urlBak, verify=False)
        time.sleep(3)
        r = session.get(url, verify=False)
    fileName = "temp/%s.png" % uuid.uuid4()
    with open(fileName, "wb") as file:
        file.write(r.content)
    return fileName


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
        cutImageName.append(save(x3,imageDir,curtImg ))
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
if __name__ == '__main__':
    image="E:\\aaaaa\\3.jpg"
    flag=judgeImageBackground(image)
    print(flag)
    imageShape = (64,64)
    cutImageName = splitImageText(image,imageShape,flag)
    print(cutImageName)