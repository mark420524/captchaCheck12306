# coding=utf-8
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imagesearch.smallervggnet import SmallerVGGNet
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os



ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
    help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
    help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output accuracy/loss plot")
args = vars(ap.parse_args())
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (67, 67, 3)
MODEL_CACHE_DIR = os.path.join(os.getcwd(), "save", "save_cache")
LOSS_PLOT_PATH=args["plot"]
# initialize the data and labels
data = []
labels = []



def load_data(fn , is_label=False):
    data_set = np.load(fn)
    texts,labels=data_set['dataset'],data_set['labels']
    texts = texts/255
    _,h,w,d=texts.shape
    texts.shape=(-1,h,w,d)

    if is_label:
        labels = to_categorical(labels)
    n = int(texts.shape[0] * 0.9)   # 90%用于训练，10%用于测试
    return (texts[:n], labels[:n]), (texts[n:], labels[n:])
 
# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
#imagePaths = sorted(list(paths.list_images(args["dataset"])))
#random.seed(42)
#random.shuffle(imagePaths)
#print(imagePaths)
# loop over the input images
#exit
"""
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    #im_file = os.path.join('', imagePath)
    image = cv2.imdecode(np.fromfile(imagePath, dtype=np.uint8),cv2.IMREAD_COLOR)
    # 直接读取图片的路径 图片读取失败
    #image = cv2.imread(im_file)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)
 
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
"""
# scale the raw pixel intensities to the range [0, 1]
(trainX, trainY),(testX, testY) = load_data(args['dataset'])  
#labels = np.array(labels)

 
# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(trainY)
 
# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
#(trainX, testX, trainY, testY) = train_test_split(data,
#    labels, test_size=0.2, random_state=42)
"""
  图像预处理  参数意义参见  https://keras.io/zh/preprocessing/image/
"""
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")
# initialize the model
print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
                            depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
"""
   编译训练, 
   优化器optimizer: Adam
        decay: float >= 0. 每次参数更新后学习率衰减值。INIT_LR / EPOCHS
   损失函数loss : sparse_categorical_crossentropy
   评估标准 Metrics:
"""
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])
 
# train the network
print("[INFO] training network...")
model_cache_path = os.path.join(MODEL_CACHE_DIR,"model_cache.model")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1,
    callbacks=[
        ModelCheckpoint(
            model_cache_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1),
        # EarlyStopping(patience=50),
        ReduceLROnPlateau(patience=10),
        CSVLogger("training.log")
        ])
# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])
 
# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# 打印训练结果集图形化
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_acc'], label='val_acc')
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(LOSS_PLOT_PATH)