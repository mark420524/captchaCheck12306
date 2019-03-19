# coding: utf-8
import pathlib

import cv2
import numpy as np
import os
from preparent import read_data
epoch_size=800
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
def read_data_bak(data_dir):
    datas = [] 
    full_name = []
    for fname in os.listdir(data_dir):
        full_image_path = os.path.join(data_dir ,fname)
        img = cv2.imdecode(np.fromfile(full_image_path, dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        #img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(img.shape)
        #img = img / 255.0
        datas.append(img)
        full_name.append(fname)
    return np.array(datas),full_name
def load_data(fn='texts.npz', to=False):
    from keras.utils import to_categorical
    data = np.load(fn)
    texts, labels = data['texts'], data['labels']
    texts = texts / 255.0
    _, h, w = texts.shape
    texts.shape = (-1, h, w, 1)
    if to:
        labels = to_categorical(labels)
    #n = int(texts.shape[0] * 0.9)   # 90%用于训练，10%用于测试
    return (texts , labels), (texts, labels)


def savefig(history, fn='loss.jpg', start=2):
    import matplotlib.pyplot as plt
    # 忽略起点
    loss = history.history['loss'][start - 1:]
    val_loss = history.history['val_loss'][start - 1:]
    epochs = list(range(start, len(loss) + start))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fn)


def main():
    from keras import models
    from keras import layers
    from keras.callbacks import ReduceLROnPlateau
    (train_x, train_y), (test_x, test_y) = load_data()
    _, h, w, _ = train_x.shape
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(h, w, 1)),
        layers.MaxPooling2D(),  # 19 -> 9
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(),  # 9 -> 4
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D(),  # 4 -> 2
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(64, activation='relu'),
        layers.Dense(80, activation='softmax'),
    ])
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # 当标准评估停止提升时，降低学习速率
    reduce_lr = ReduceLROnPlateau(verbose=1)
    history = model.fit(train_x, train_y, epochs=epoch_size,
                        validation_data=(test_x, test_y),
                        callbacks=[reduce_lr])
    savefig(history, start=10)
    model.save('model.v1.0.h5', include_optimizer=False)


def load_data_v2():
    (train_x, train_y), (test_x, test_y) = load_data(to=True)
    # 这里是统计学数据
    (train_v2_x, train_v2_y), (test_v2_x, test_v2_y) = load_data('texts.v2.npz')
    # 合并
    train_x = np.concatenate((train_x, train_v2_x))
    train_y = np.concatenate((train_y, train_v2_y))
    test_x = np.concatenate((test_x, test_v2_x))
    test_y = np.concatenate((test_y, test_v2_y))
    return (train_x, train_y), (test_x, test_y)


def acc(y_true, y_pred):
    import keras.backend as K
    return K.cast(K.equal(K.argmax(y_true + y_pred, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def main_v19():     # 1.9
    from keras import models
    from keras.callbacks import ReduceLROnPlateau
    (train_x, train_y), (test_x, test_y) = load_data_v2()
    model = models.load_model('model.v1.0.h5')
    model.compile(optimizer='RMSprop',
                  loss='categorical_hinge',
                  metrics=[acc])
    reduce_lr = ReduceLROnPlateau(verbose=1)
    history = model.fit(train_x, train_y, epochs=epoch_size,
                        validation_data=(test_x, test_y),
                        callbacks=[reduce_lr])
    savefig(history)
    model.save('model.v1.9.h5', include_optimizer=False)


def main_v20():
    from keras import models
    from keras import layers
    from keras.callbacks import ReduceLROnPlateau
    (train_x, train_y), (test_x, test_y) = load_data()
    _, h, w, _ = train_x.shape
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(h, w, 1)),
        layers.MaxPooling2D(),  # 19 -> 9
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),  # 9 -> 4
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),  # 4 -> 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(),  # 2 -> 1
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.25),
        layers.Dense(64, activation='relu'),
        layers.Dense(80, activation='softmax'),
    ])
    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=epoch_size,
              validation_data=(test_x, test_y))
    (train_x, train_y), (test_x, test_y) = load_data_v2()
    model.compile(optimizer='rmsprop',
                  loss='categorical_hinge',
                  metrics=[acc])
    reduce_lr = ReduceLROnPlateau(verbose=1)
    history = model.fit(train_x, train_y, epochs=100,
                        validation_data=(test_x, test_y),
                        callbacks=[reduce_lr])
    savefig(history)
    # 保存，并扔掉优化器
    model.save('model.v2.0.h5', include_optimizer=False)


def predict(texts):
    from keras import models
    model = models.load_model('model.v1.0.h5')
    texts = texts / 255.0
    _, h, w = texts.shape
    texts.shape = (-1, h, w, 1)
    labels = model.predict(texts)
    return labels


def _predict():
    texts = np.load('data.npy')
    labels = predict(texts)
    np.save('labels.npy', labels)


def show():
    texts = np.load('data.npy')
    labels = np.load('labels.npy')
    labels = labels.argmax(axis=1)
    pathlib.Path('classify').mkdir(exist_ok=True)
    for idx, (text, label) in enumerate(zip(texts, labels)):
        # 使用聚类结果命名
        fn = f'classify/{label}.{idx}.jpg'
        cv2.imwrite(fn, text)
def test_predict(data_dir):
    
    data,full_name  = read_data_bak(data_dir)
    #print(len(data))
    #print(lables)
    labels = predict( data )
    labels = labels.argmax(axis=1)
    count = 0
    for label in labels:
        print("{}={}".format(full_name[count],label_dict[label]))
        #print(label_dict[label])
        count=count+1

if __name__ == '__main__':
    #main()
    main_v20()
    #_predict()
    #show()
    #data_dir = "E:\\aaaaa\\3333\\"
    #test_predict(data_dir)   
    #print(labels)
