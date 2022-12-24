import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import tensorflow as tf
import keras
import cv2
import seaborn as sns
import random
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception,VGG16,ResNet50
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
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
x_train, x_val, y_train, y_val = train_test_split(X_img_normalize, y_label, test_size = 0.2)

#CNN
# Defining base model using Xception module from Keras
training_shape = (71, 71, 3)
base_model = Xception(include_top=False,weights='imagenet',input_shape = training_shape)
for layer in base_model.layers:
    layer.trainable = True                         # Training all layers from scratch

#Adding layers at end
n_classes = 7
model = base_model.output
model = Flatten()(model)
model = Dense(128)(model)
model = Dropout(0.5)(model)
model = BatchNormalization()(model)
model = Activation('relu')(model)
output = Dense(n_classes, activation='softmax')(model)
model = Model(inputs=base_model.input, outputs=output)

# Compiling the model
optimizer = Adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer, 
              metrics=['accuracy'])

# Defining callback Methods
n_epoch = 30

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, 
                           mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, 
                              verbose=1, mode='auto')

# Fitting the model
history = model.fit(x_train,
                    y_train,
                    epochs=n_epoch,
                    callbacks=[reduce_lr,early_stop],
                    validation_data=(x_val,y_val)
                   )
'''
# Plotting the results on Graph
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
'''
'''
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
#y_pred = y_pred.round()
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

x_test_normalize = img_test.astype('float32') / 255.0

df_submit = pd.DataFrame([], columns=['Id', 'Label'])
df_submit['Id'] = [f'{i:04d}' for i in range(len(x_test_normalize))]
df_submit['Label'] = model.predict(x_test_normalize)

df_submit.to_csv('data/predict.csv', index=None)
'''