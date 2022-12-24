#keras.utils: 做one-hot encoding用
#sklearn.model_selection: 分割訓練集和測試集
#os: 用來建立檔案、刪除檔案
#PIL: (圖像處理庫)匯入圖像
#seed: 設定種子，使每次隨機產生的資料有相同結果。可將數字改成自己的學號(或其他數字)
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import os
from PIL import Image
np.random.seed(408570344)

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
pd.Categorical(lesion_type_dict).codes

load_img = pd.read_csv('data/project3_train.csv')

#iloc選取特定範圍，讀取種類編號
X_img , y_label = load_img.iloc[: , :-1].values , load_img.iloc[: , -1].values

#將串列轉成矩陣
X_img_train = np.asarray(X_img.tolist())

#將一維的數據，轉換成三維(長*寬*RGB三色)
X_img_train=X_img_train.reshape(X_img_train.shape[0],28,28,3)

#檢查學習資料的照片數量、尺寸大小、維度
print("train data:",'images:',X_img_train.shape," labels:",y_label.shape) 

#標準化: 同除255(因為image的數字是0~255)
X_img_train_normalize = X_img_train.astype('float32') / 255.0

#使用np_utils.to_categorical()傳入各參數的label標籤欄位，再執行OneHot encoding (轉成0或1的組合)
y_label_train_OneHot = np_utils.to_categorical(y_label)

x_train, x_test, y_train, y_test = train_test_split(X_img_train_normalize, y_label, test_size = 0.3)

#匯入keras中的Sequential、layers模組(Dense、 Dropout、 Activation、 Flatten、Conv2D、 MaxPooling2D、 ZeroPadding2D)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# 建立簡單的線性執行的模型
model = Sequential()
# 建立卷積層，filter=32,即 output space 的深度, Kernal Size: 3x3, activation function 採用 relu
model.add(Conv2D(filters = 256,
                 kernel_size = (3, 3),
                 activation = 'relu',
                 input_shape = [28,28,3],
                 padding = 'same'))
# 建立卷積層，filter=64,即 output size, Kernal Size: 3x3, activation function 採用 relu
#model.add(Conv2D(64, (3, 3), activation='relu'))
# 建立池化層，池化大小=2x2，取最大值
model.add(Conv2D(256, (3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(1028, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(1028, (3, 3), activation='relu', padding = 'same'))
model.add(Conv2D(1028, (3, 3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten層把多維的輸入一維化，常用在從卷積層到全連接層的過渡。
model.add(Flatten())
# 全連接層: 128個output
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
# Dropout層隨機斷開輸入神經元，用於防止過度擬合，斷開比例:0.5
model.add(Dropout(0.25))
# 使用 softmax activation function，將結果分類
model.add(Dense(7, activation='softmax'))

# 編譯: 選擇損失函數、優化方法及成效衡量方式
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 進行訓練, 訓練過程會存在 train_history 變數中
train_history = model.fit(X_img_train_normalize, y_label_train_OneHot,
                          validation_split = 0.2,
                          batch_size = 10,
                          epochs = 12,
                          verbose = 1)
'''
# 顯示損失函數、訓練成果(分數)
score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
# Design your CNN model
# 使用最後的模型進行測試資料預測
load_test_img = pd.read_csv('data/project3_test.csv')
img_test = load_test_img.values
x_test=img_test.reshape(img_test.shape[0],28,28,3)
x_test_normalize = x_test.astype('float32') / 255.0
df_submit = pd.DataFrame([], columns=['Id', 'Label'])
df_submit['Id'] = [f'{i:04d}' for i in range(len(x_test_normalize))]
df_submit['Label'] = np.argmax(model.predict(x_test_normalize), axis=-1)
df_submit.to_csv('data/predict.csv', index=None)
