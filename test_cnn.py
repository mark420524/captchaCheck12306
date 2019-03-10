#coding=utf-8

import os
#图像读取库
from PIL import Image
#矩阵运算库
import numpy as np
import tensorflow as tf

model_dir = "save/"
model_path = "text_model"
label_dict={
    0: '中国结', 1: '仪表盘', 2: '公交卡', 3: '冰箱', 4: '创可贴', 
    5: '刺绣', 6: '剪纸', 7: '印章', 8: '卷尺', 9: '双面胶', 10: '口哨', 
    11: '啤酒', 12: '安全帽',13: '开瓶器', 14: '手掌印', 15: '打字机', 
    16: '护腕', 17: '拖把', 18: '挂钟', 19: '排风机', 20: '文具盒', 
    21: '日历', 22: '本子', 23: '档案袋', 24: '棉棒', 25:'樱桃', 26: '毛线', 
    27: '沙包', 28: '沙拉', 29: '海报', 30: '海苔', 31: '海鸥',32: '漏斗', 
    33: '烛台', 34: '热水袋', 35: '牌坊', 36: '狮子', 37: '珊瑚', 38: '电子秤', 
    39: '电线', 40: '电饭煲', 41: '盘子', 42: '篮球', 43: '红枣', 44: '红豆', 
    45: '红酒', 46: '绿豆', 47: '网球拍', 48: '老虎', 49: '耳塞', 50: '航母', 
    51: '苍蝇拍', 52: '茶几', 53: '茶盅', 54: '药片', 55: '菠萝', 56: '蒸笼', 
    57: '薯条', 58: '蚂蚁', 59: '蜜蜂', 60: '蜡烛', 61: '蜥蜴', 62: '订书机', 
    63: '话梅', 64: '调色板', 65: '跑步机', 66: '路灯', 67: '辣椒酱', 68: '金字塔', 
    69: '钟表', 70: '铃铛', 71: '锅铲', 72: '锣', 73: '锦旗', 74: '雨靴', 
    75: '鞭炮', 76: '风铃', 77: '高压锅', 78: '黑板', 79: '龙舟'}

def read_data(full_image_path,image_shape=(64, 64)):
    image = Image.open(full_image_path)
    if image_shape and image.size != image_shape[:2]:
        image = image.resize(image_shape[:2])
    data = np.array(image) 
    datas = []
    labels = []
    labels.append(15)
    datas.append(data)
    return np.array(datas),np.array(labels)
def read_label( label_path="label.txt"):
    if os.path.exists(label_path):
        with open(label_path,encoding="utf-8") as files:
            for lines in files:
                # split 默认已空格拆分字符串
                lable_name,id = lines.strip().split()
                label_dict[int(id)]=lable_name
                
    return label_dict
num_classes = len(label_dict.keys())
        
#print(num_classes)
datas_placeholder = tf.placeholder(tf.float32, [None, 64, 64, 3])
labels_placeholder = tf.placeholder(tf.int32, [None])

# 存放DropOut参数的容器，训练时为0.25，测试时为0
dropout_placeholdr = tf.placeholder(tf.float32)


# 定义卷积层, 20个卷积核, 卷积核大小为5，用Relu激活
conv0 = tf.layers.conv2d(datas_placeholder, 20, 3, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [1, 1])

# 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
conv1 = tf.layers.conv2d(pool0, 40, 3, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [1, 1])

# 定义卷积层, 40个卷积核, 卷积核大小为4，用Relu激活
conv2 = tf.layers.conv2d(pool1, 60, 2, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool2 = tf.layers.max_pooling2d(conv2, [2, 2], [1, 1])

conv3 = tf.layers.conv2d(pool2, 80, 2, activation=tf.nn.relu)
# 定义max-pooling层，pooling窗口为2x2，步长为2x2
pool3 = tf.layers.max_pooling2d(conv3, [2, 2], [1, 1])

# 将3维特征转换为1维向量
flatten = tf.layers.flatten(pool3)

# 全连接层，转换为长度为100的特征向量
fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)

# 加上DropOut，防止过拟合
dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)

# 未激活的输出层
logits = tf.layers.dense(dropout_fc, num_classes)

predicted_labels = tf.arg_max(logits, 1)
 
# 利用交叉熵定义损失
#losses = tf.nn.softmax_cross_entropy_with_logits(
#    labels=tf.one_hot(labels_placeholder, num_classes),
#    logits=logits
#)
# 平均损失
#mean_loss = tf.reduce_mean(losses)

# 定义优化器，指定要优化的损失函数
#optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)
# 用于保存和载入模型
saver = tf.train.Saver()
def test_model(full_image_path):
    
    with tf.Session() as sess:
        
        
        #model_file=tf.train.latest_checkpoint(model_dir)
        #saver.restore(sess,model_file)
        saver = tf.train.import_meta_graph(model_dir + model_path+'.meta')
        saver.restore(sess, model_dir + model_path  )
        datas,labels = read_data(full_image_path)
        test_feed_dict = {
            datas_placeholder: datas,
            labels_placeholder:labels,
            dropout_placeholdr: 0
        }
        
        val_acc = sess.run(predicted_labels, feed_dict=test_feed_dict)
        return val_acc
if __name__ == "__main__":
    #read_label()
    #print(label_dict)
    full_image_dir = "E:\\aaaaa\\1111\\"
    for file_name in os.listdir(full_image_dir):
        full_image_path = os.path.join(full_image_dir,file_name)
        a=test_model(full_image_path)
        print("filename {} test datas  {}\t validate labels: {}".format(file_name,a, label_dict[int(a)]))
        
