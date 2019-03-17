import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import hashlib
X = np.linspace(-2, 6, 200)
np.random.shuffle(X)
Y = 0.5 * X + 2 + 0.15 * np.random.randn(200,) 
 
# plot data
plt.scatter(X, Y)
plt.show()
 
X_train, Y_train = X[:160], Y[:160]     # train first 160 data points
X_test, Y_test = X[160:], Y[160:]       # test remaining 40 data points

model = Sequential()
model.add(Dense(output_dim=1,input_dim=1))
model.compile(loss='mse', optimizer='sgd')
model.compile(loss='mse', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))


print("welcome python hashlib world!")
# 256
hash_256 = hashlib.sha256()
hash_str = "gxchain"
hash_256.update(hash_str.encode('utf-8'))
hash_256_value = hash_256.hexdigest()
obj = hashlib.new('ripemd160', hash_str.encode('utf-8'))
ripemd_160_value = obj.hexdigest()
print("sha256:", hash_256_value)  # 16进制
print("ripemd160 :", ripemd_160_value)