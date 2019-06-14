from keras.models import Sequential
import random
model = Sequential()
import numpy as np
images=np.zeros((100,64,64,3))
for i in range(100):
    images[i]=np.random.randint(low=0,high=255,size=(64,64,3))
    
import matplotlib.pyplot as plt
plt.imshow(images[0])
images=images.reshape(images.shape[0],-1)

from keras.layers import Dense

model.add(Dense(units=3, activation='tanh', input_dim=12288))
model.add(Dense(units=4, activation='tanh'))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(images, y_train, epochs=5, batch_size=1)

y_train=np.zeros((100,10))
for i in range(100):
    n=random.randint(0,9)
    y_train[i][n]=1
    
model.fit(images, y_train, epochs=5, batch_size=1)

x_test=np.random.randint(low=0,high=255,size=(1,12288))
y_test=np.zeros((1,10))
classes = model.predict(x_test, batch_size=1)
print(classes)
