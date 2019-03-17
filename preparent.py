

import cv2
import numpy as np

import os


# 数据文件夹
data_dir = "resize_words"

# 模型文件路径

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
            img = cv2.imdecode(np.fromfile(full_image_path, dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
            #img = cv2.imread(full_image_path, cv2.IMREAD_GRAYSCALE)
            print(img.shape)
            #image = Image.open(full_image_path)
            #data = np.array(image)  
            #label = int(fname.split("_")[0])
            datas.append(img)
            if need_save:
                label_name_dict[fname] = count_dir
                label_dict[count_dir] = fname
                labels.append(count_dir)
            else:
                labels.append(label_name_dict[fname])
        count_dir = count_dir+1
    #datas = np.array(datas)
    #labels = np.array(labels)
    
    if need_save:
        with open(label_path,"w",encoding="utf-8") as files:
            #count_index=0
            for key in label_name_dict:
                # split 默认已空格拆分字符串
                files.write("%s %s\n" % (key ,label_name_dict[key] ))
                #count_index = count_index+1
                #lable_name,id = lines.strip().split()
                #label_object[lable_name]=int(id)
    #print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return datas, labels

def load_data(path='texts.npz'):
    if not os.path.isfile(path):
        texts, labels = read_data(data_dir)
        np.savez(path, texts=texts, labels=labels)
    f = np.load(path)
    return f['texts'], f['labels']
if __name__ == '__main__':
    #read_data(data_dir)
    texts, labels = load_data()
    print(len(texts))
    print(len(set(labels)))
    print(texts.shape)
    print(labels.shape)
