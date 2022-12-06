from asyncio.windows_events import NULL
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, GlobalAveragePooling1D, LeakyReLU
from keras.layers import BatchNormalization, Conv1D, MaxPooling1D, MaxPooling2D, Conv2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras import activations

import tensorflow as tf
import matplotlib.pyplot as plt

#tf.debugging.set_log_device_placement(True)

# df = pd.read_csv('interpolateData.csv', header = None) 
# test_df = pd.read_csv('interpolateTest.csv', header = None) 
df = pd.read_csv('4wayData0123.csv') 
test_df = pd.read_csv('4wayTest0123.csv') 
df = df.drop(["Group"], axis=1)
test_df = test_df.drop(["Group"], axis=1)
# df = df.drop([0], axis=1)
# test_df = test_df.drop([0], axis=1)

def buildTrain(train, pastDay, futureDay):#把資料轉成20*16維度
  X_train, Y_train = [], []
  for i in range(0, train.shape[0], pastDay):
    X_train.append(np.array(train.iloc[i:i+pastDay]))
    Y_train.append(np.array(train.iloc[i:i+futureDay]["Dir"]))
  return np.array(X_train), np.array(Y_train)

def normalize(train):
  temp = train['Dir']
  # temp = train[65]
  # del train[65]
  train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  train_norm['Dir'] = temp #預測Dir
  # train_norm[65] = temp #預測Dir
  return train_norm

def shuffle(X,Y):
  np.random.seed(10) #打亂資料
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  #print("X randomList",X[randomList])
  return X[randomList], Y[randomList]

def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val

#%%

train = df
test = test_df
train = normalize(df)
test = normalize(test_df)

trainX, trainY = buildTrain(train, pastDay=20, futureDay=1)
testX, testY = buildTrain(test, pastDay=20, futureDay=1)
trainX, trainY = shuffle(trainX, trainY)
testX, testY = shuffle(testX, testY)

trainY = to_categorical(trainY)
testY = to_categorical(testY)

#X_train, Y_train, X_val, Y_val = splitData(trainX, trainY, 0.1) #分割資料10%做為測試資料
X_train = trainX
Y_train = trainY
X_val = testX
Y_val = testY

#%%
X_train = X_train[:,:,:-1]
X_val = X_val[:,:,:-1]

#%%

X_train.shape
#X_train[:10]
#%%
Y_train.shape
#Y_train[:10]

print('X_train.shape ', X_train.shape[0],X_train.shape[1],X_train.shape[2])
print('Y_train.shape ',Y_train.shape)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1).astype('float32')
X_val = X_val.reshape(X_val.shape[0],X_train.shape[1],X_train.shape[2],1).astype('float32')

print(X_train.shape)

#%%

model = Sequential()
# model.add(Conv1D(filters=16, kernel_size=(2), padding='same', input_shape = (8,8), activation='relu'))
# model.add(MaxPooling1D(pool_size=(2)))
# model.add(Conv1D(filters=36, kernel_size=(2), padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=(2)))
# model.add(Conv1D(filters=64, kernel_size=(2), padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=(2)))
# # model.add(Dropout(0.2)) #dropout說先拿掉 #1d改2d kernelsize用3x3 filter可以嘗試多一點 特徵、svm、scikit-learn、decision tree
# model.add(Flatten())
# model.add(Dense(32,activation='relu'))
# model.add(Dense(4,activation='softmax'))
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
# model.summary()

# model.add(Conv2D(filters=48,kernel_size=(3,3),padding='same',input_shape=(20,16,1),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Conv2D(filters=48,kernel_size=(3,3),padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3, 3), input_shape=(20,16,1), padding='same',activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same',))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(4,activation='softmax'))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.summary()
#%%

Y_train.shape

#%%

#Y_train = Y_train[:,np.newaxis]
#Y_val = Y_val[:,np.newaxis]

#%%
print('X_train.shape ', X_train.shape)
print('Y_train.shape ',Y_train.shape)

callback = [EarlyStopping(monitor="loss", patience=20, verbose=1, mode="auto")] #monitor="loss"
history = model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_data=(X_val, Y_val), callbacks = callback) #epochs=1000 batch_size = 64
scores = model.evaluate(X_val, Y_val, verbose=0)
a = scores[1]*100
print("Model Accuracy: %.2f%%" % (a))
temp = "Model Accuracy: " + str(a) + "\n\n"
path = 'Accuracy.txt'
with open(path, 'a') as f:
  f.write(temp)

model.save('my_modelStatic.h5')

with open(path, 'a') as f:
  model.summary(print_fn=lambda x: f.write(x + '\n'))


#%%

model.summary()
from numpy import argmax
# 預測結果
x = np.argmax(model.predict(X_val), axis=-1)
#x = model.predict_classes(X_val)


#%%
y_true = argmax(Y_val, axis=1)

y_pred = x

cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
print(cm)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

def plot_confusion_matrix(cm, class_names):
    """
    產生一張圖表示的 confusion matrix

    Args:
    cm (shape = [n, n]): 傳入 confusion matrix
    class_names (shape = [n]): 傳入類別
    """
    # 標準化 confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    figure = plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    tick_index = np.arange(len(class_names))
    # matplotlib 3.1.1 bug，如果不設定 ylim 在 [-0.5 ~ 2.5]，圖片 y 軸範圍會被縮小成 [0 ~ 2]
    #plt.ylim([-1, 5])
    # Y 軸顯示類別名稱
    plt.yticks(tick_index, class_names)
    # X 軸顯示類別名稱，並將類別名稱旋轉 45 度 (避免文字重疊)
    plt.xticks(tick_index, class_names, rotation=45)
    # 在圖片右邊產生一條顏色刻度條
    plt.colorbar()

    # 在每一格 confusion matrix 輸入預測百分比
    threshold = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # 如果格內背景顏色太深使用白色文字顯示，反之使用黑色文字
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # 將圖片的位置進行調整，避免 x 或 y 軸的文字被遮擋
    plt.tight_layout()
    return figure

img = plot_confusion_matrix(cm, ['ges0','ges1','ges2','ges3','ges4'])
img.savefig('CM.png')
# img.show()

# for idx in range(40):
#     print('* data', X_val[idx])
#     print()
#     print('* predict', x[idx])
#     print()
#     print()



