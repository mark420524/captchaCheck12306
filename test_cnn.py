#coding=utf-8

import os
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
import tensorflow as tf

model_dir = "save/"
model_path = "text_model"
label_dict={}

def read_data(full_image_path):
    image = Image.open(full_image_path)
    data = np.array(image) / 255.0
    return np.array(data)
def read_label( label_path="label.txt"):
    if os.path.exists(label_path):
        with open(label_path,encoding="utf-8") as files:
            for lines in files:
                # split 默认已空格拆分字符串
                lable_name,id = lines.strip().split()
                label_dict[int(id)]=lable_name
                
    return label_dict
def test_model(full_image_path):
    with tf.Session() as sess:
        
        num_classes = len( label_dict.keys() )
        
        print("测试图片文字")
        datas_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 3])
        labels_placeholder = tf.placeholder(tf.int32, [None])

        # 存放DropOut参数的容器，训练时为0.25，测试时为0
        dropout_placeholdr = tf.placeholder(tf.float32)

        # 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
        conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
        # 定义max-pooling层，pooling窗口为2x2，步长为2x2
        pool0 = tf.layers.max_pooling2d(conv0, [1, 1], [1, 1])

        # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
        conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
        # 定义max-pooling层，pooling窗口为2x2，步长为2x2
        pool1 = tf.layers.max_pooling2d(conv1, [1, 1], [1, 1])

        # 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
        conv2 = tf.layers.conv2d(pool0, 80, 4, activation=tf.nn.relu)
        # 定义max-pooling层，pooling窗口为2x2，步长为2x2
        pool2 = tf.layers.max_pooling2d(conv1, [1, 1], [1, 1])

        # 将3维特征转换为1维向量
        flatten = tf.layers.flatten(pool2)

        # 全连接层，转换为长度为100的特征向量
        fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

        # 加上DropOut，防止过拟合
        dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

        # 未激活的输出层
        logits = tf.layers.dense(dropout_fc, num_classes)

        predicted_labels = tf.arg_max(logits, 1)
         
        saver = tf.train.import_meta_graph(model_dir + model_path+'.meta')
        saver.restore(sess, model_dir + model_path  )
        test_feed_dict = {
            datas_placeholder: read_data(full_image_path),
            labels_placeholder:[1],
            dropout_placeholdr: 0
        }
        
        label_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        return label_val
if __name__ == "__main__":
    read_label()
    full_image = "E:\\aaaaa\\1111\\1.png"
    print(test_model(full_image))
