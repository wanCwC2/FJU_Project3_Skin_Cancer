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

load_img = pd.read_csv('data/project3_train.csv')
img_test = pd.read_csv('data/project3_test.csv')

X_img , y_label = load_img.iloc[: , :-1].values , load_img.iloc[: , -1].values

#將串列轉成矩陣
X_img_train = np.asarray(X_img.tolist())

#將一維的數據，轉換成三維(長*寬*RGB三色)
X_img_train=X_img_train.reshape(X_img_train.shape[0],28,28,3)

#標準化: 同除255(因為image的數字是0~255)
X_img_train_normalize = X_img_train.astype('float32') / 255.0

#使用np_utils.to_categorical()傳入各參數的label標籤欄位，再執行OneHot encoding (轉成0或1的組合)
y_label_train_OneHot = np_utils.to_categorical(y_label)

X_img_normalize = X_img.astype('float32') / 255.0

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report,confusion_matrix

#建立隨機森林的實體
#model_RF = RandomForestClassifier()
#分割學習資料集與驗證資料集
X_train, X_test, y_train, y_test = train_test_split(X_img_normalize, y_label, test_size = 0.2)

# Design your RandomForest model
# XGBoost
from xgboost import XGBClassifier
from xgboost import XGBRegressor
'''
params = { 'max_depth': range (2, 15, 3),
           'learning_rate': [0.01, 0.1, 0.5, 1, 5, 10],
           'n_estimators': range(80, 500, 50),
           'colsample_bytree': [0.5, 1, 3, 6, 10],
#           'min_child_weigh': range(1, 9, 1),
           'subsample': [0.5, 0.7, 0.9, 1.5, 2]}
from sklearn.model_selection import GridSearchCV
model = XGBClassifier()
clf = GridSearchCV(estimator = model,
                   param_grid = params,
                   scoring = 'neg_log_loss')
clf.fit(X_train, y_train)
print("Best parameters:", clf.best_params_)
# Best parameters: {'colsample_bytree': 1, 'learning_rate': 0.01, 'max_depth': 2, 'n_estimators': 380, 'subsample': 0.9}
print(clf.best_estimator_)
# XGBClassifier(colsample_bytree=1, learning_rate=0.01, max_depth=2, n_estimators=380, subsample=0.9)
'''

#CNN
from keras.models import Sequential
#from keras.models import Conv2D
#from keras.layers.convolutional import Conv2D
import keras

#model = clf.best_estimator_
#model = XGBClassifier(colsample_bytree=1, learning_rate=0.01, max_depth=2, n_estimators=380, subsample=0.9)
#model = XGBClassifier()
#model = XGBRegressor()

model = Sequential()
model.add(keras.layers.Conv2D(filters=7, kernel_size=5, strides=1, padding="same", activation="relu",input_shape=[28, 28, 1]))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Conv2D(filters=7, kernel_size=3, strides=1, padding="same", activation="relu"))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dense(50, activation="relu"))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(10, activation="softmax"))


model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
#y_pred = y_pred.round()
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

x_test_normalize = img_test.astype('float32') / 255.0

df_submit = pd.DataFrame([], columns=['Id', 'Label'])
df_submit['Id'] = [f'{i:04d}' for i in range(len(x_test_normalize))]
df_submit['Label'] = model.predict(x_test_normalize)

df_submit.to_csv('data/predict.csv', index=None)