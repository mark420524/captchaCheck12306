# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
    @staticmethod
    def build(width,height,depth,classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        # CONV => RELU => POOL
        """
         卷积层 二维卷积Conv2D 二维作用于图片
         Conv2D(filters, kernel_size, strides=(1, 1), padding=’valid’)
         filters：卷积核的个数。

         kernel_size：卷积核的大小。

         strdes：步长，二维中默认为(1, 1)，一维默认为1。

         Padding：补“0”策略，’valid‘指卷积后的大小与原来的大小可以不同，’same‘则卷积后大小与原来大小一致。
        """
        model.add(Conv2D(32, (3, 3), padding="same",
            input_shape=inputShape))
        """
         激活层：对上一层的输出应用激活函数。
         Activation(activation)
         activation 激活函数名称详见api
        """
        model.add(Activation("relu"))
        """
         批量标准化层
         axis: 整数，需要标准化的轴 （通常是特征轴）。 
         例如，在 data_format= channels_first  的 Conv2D 层之后， 在 BatchNormalization 中设置 axis=1
         使用tk backend image_data_format=channels_last  so axis=-1
        """
        model.add(BatchNormalization(axis=chanDim))
        """
          池化层 
          最大统计量池化和平均统计量池化也有三种 MaxPooling1D、MaxPooling2D、MaxPooling3D
          和AveragePooling1D、AveragePooling2D、AveragePooling3D
          MaxPooling(pool_size=(2,2), strides=None, padding=’valid’)
          pool_size：长度为2的整数tuple，表示在横向和纵向的下采样因子，一维则为纵向的下采样因子。
          padding：和卷积层的padding一样
        """
        model.add(MaxPooling2D(pool_size=(3, 3)))
        """
         Dropout层：对上一层的神经元随机选取一定比例的失活
         Dropout(rate)  
         rate: 失活比例,0-1浮点数 防止过拟合
        """
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))


        # add 256 filter size 
        model.add(Conv2D(256, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        # add 512 filter size 
        model.add(Conv2D(512, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(512, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))



        # first (and only) set of FC => RELU layers
        """
         Flatten层：将一个维度大于或等于3的高维矩阵，“压扁”为一个二维矩阵
        """
        model.add(Flatten())
        """
         全连接层  实现对神经网络里的神经元激活  
         Dense（units, activation=’relu’, use_bias=True）
         units: 全连接层输出的维度，即下一层神经元的个数。

         activation：激活函数，默认使用Relu。

         use_bias：是否使用bias偏置项。
        """
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
 
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
 
        # return the constructed network architecture
        return model