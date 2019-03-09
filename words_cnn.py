#coding=utf-8

import os
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
import tensorflow as tf


# 数据文件夹
data_dir = "resize_words"
# 训练还是测试
train = True
# 模型文件路径
model_path = "save/text_model"
label_name_dict={}
label_dict={}
# 从文件夹读取图片和标签到numpy数组中
# 标签信息在文件名中，例如1_40.jpg表示该图片的标签为1
def read_data(data_dir,label_path="label.txt"):
    datas = []
    labels = []
    fpaths = []
    count_dir = 0
    need_save = True
    if os.path.exists(label_path):
        need_save = False
        with open(label_path,encoding="utf-8") as files:
            for lines in files:
                # split 默认已空格拆分字符串
                lable_name,id = lines.strip().split()
                label_name_dict[lable_name]=id
                label_dict[int(id)]=lable_name
    
    for fname in os.listdir(data_dir):
        for file_name in os.listdir(os.path.join(data_dir, fname)):
            full_image_path = os.path.join(data_dir,fname,file_name)
            #fpath = os.path.join(data_dir, fname)
            fpaths.append(full_image_path)
            image = Image.open(full_image_path)
            data = np.array(image) / 255.0
            #label = int(fname.split("_")[0])
            datas.append(data)
            if need_save:
                label_name_dict[fname] = count_dir
                label_dict[count_dir] = fname
                labels.append(count_dir)
            else:
                labels.append(label_name_dict[fname])
        count_dir = count_dir+1
    datas = np.array(datas)
    labels = np.array(labels)
    
    if need_save:
        with open(label_path,"w",encoding="utf-8") as files:
            #count_index=0
            for key in label_name_dict:
                # split 默认已空格拆分字符串
                files.write("%s %s\n" % (key ,label_name_dict[key] ))
                #count_index = count_index+1
                #lable_name,id = lines.strip().split()
                #label_object[lable_name]=int(id)
    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels


fpaths, datas, labels = read_data(data_dir )
# 计算有多少类图片
num_classes = len(set(labels))


# 定义Placeholder，存放输入和标签
datas_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 3])
labels_placeholder = tf.placeholder(tf.int32, [None])

# 存放DropOut参数的容器，训练时为0.25，测试时为0
dropout_placeholdr = tf.placeholder(tf.float32)

# 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
conv0 = tf.layers.conv2d(datas_placeholder, 20, 8, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])

# 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
conv1 = tf.layers.conv2d(pool0, 40, 6, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])

# 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
conv2 = tf.layers.conv2d(pool1, 60, 4, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [2, 2])

conv3 = tf.layers.conv2d(pool2, 80, 2, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool3 = tf.layers.max_pooling2d(conv3, [2, 2], [2, 2])

# 将3维特征转换为1维向量
flatten = tf.layers.flatten(pool3)

# 全连接层，转换为长度为100的特征向量
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

# 加上DropOut，防止过拟合
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

# 未激活的输出层
logits = tf.layers.dense(dropout_fc, num_classes)

predicted_labels = tf.argmax(logits, 1)


# 利用交叉熵定义损失
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, num_classes),
    logits=logits
)
# 平均损失
mean_loss = tf.reduce_mean(losses)

# 定义优化器，指定要优化的损失函数
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)


# 用于保存和载入模型
saver = tf.train.Saver()

with tf.Session() as sess:

    if train:
        print("训练模式")
        # 如果是训练，初始化参数
        sess.run(tf.global_variables_initializer())
        # 定义输入和Label以填充容器，训练时dropout为0.25
        train_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0.25
        }
        for step in range(150):
            _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)

            if step % 10 == 0:
                print("step = {}\tmean loss = {}".format(step, mean_loss_val))
        saver.save(sess, model_path)
        print("训练结束，保存模型到{}".format(model_path))
    else:
        print("测试模式")
        # 如果是测试，载入参数
        saver.restore(sess, model_path)
        print("从{}载入模型".format(model_path))
        # label和名称的对照关系
        # 定义输入和Label以填充容器，测试时dropout为0
        test_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder: labels,
            dropout_placeholdr: 0
        }
        predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
        # 真实label与模型预测label
        for fpath, real_label, predicted_label in zip(fpaths, labels, predicted_labels_val):
            # 将label id转换为label名
            
            real_label_name = label_dict[int(real_label)]
            predicted_label_name = label_dict[predicted_label]
            print("{}\t{} => {}".format(fpath, real_label_name, predicted_label_name))


