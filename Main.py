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
img_test = pd.read_csv('data/project3_test.csv')

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

#使用np_utils.to_categorical()傳入各參數的label標籤欄位，再執行OneHot encoding (轉成0或1的組合)
y_label_train_OneHot = np_utils.to_categorical(y_label)

X_img_normalize = X_img.astype('float32') / 255.0