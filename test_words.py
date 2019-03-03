from model.densenet import DenseNet
from PIL import Image,ImageFile

import numpy as np
import time
import shutil
import os


n_classes = 80 
image_shape = (64,64,3)
text_model_weight = "saves/DenseNet-BC_k=12_d=40.weight"


def load_model():
    text_model = DenseNet(classes=n_classes, input_shape=image_shape, depth=40, growth_rate=12, bottleneck=True,
                          reduction=0.5, dropout_rate=0.0, weight_decay=1e-4)
    text_model.load_weights(text_model_weight)
    return text_model
def load_label_dict():
    # 读取类别名称
    label_dict = {}
    with open("label.txt", encoding="utf-8") as file:
        for line in file:
            class_name, id = line.strip().split()
            label_dict[int(id)] = class_name
    return label_dict
def image_load(image_path,image_shape):
    if isinstance(image_path, str):
        raw_image = Image.open(image_path)
    if image_shape and raw_image.size != image_shape[:2]:
        raw_image = raw_image.resize(image_shape[:2])
    return raw_image
    
def image_test(text_model,image_model,label_dict):
    image_dir="E:\\aaaaa\\1111\\5f753abb-bd22-4dcc-94a9-2b05da6c5c0a.png"
    text_image=image_load(image_dir,image_shape)
    texts = np.array([np.asarray(text_image)])
    #print(label_dict)
    # 模型输出
    text_predict = text_model.predict(texts)
    #print(text_predict)
    # 预测结果
    text_result = np.argmax(text_predict, 1)
    print(text_result)
    print(label_dict[text_result[0]])
if __name__ == '__main__':
    text_model = load_model()
    label_dict = load_label_dict()
    image_test(text_model,image_shape,label_dict)