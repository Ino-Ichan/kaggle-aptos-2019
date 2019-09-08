import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import math
import cv2

from sklearn.metrics import cohen_kappa_score

import matplotlib.pyplot as plt

import gc
from tqdm import tqdm

import os
import scipy as sp
from functools import partial


# basic
import keras
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Dropout, BatchNormalization
from keras.layers import Flatten, GlobalAveragePooling2D, GlobalMaxPool2D, Softmax
from keras import optimizers
from keras.utils import Sequence

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from efficientnet.keras import EfficientNetB5, EfficientNetB2

from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

import imgaug.augmenters as iaa
import imgaug as ia

from nn_generator import TrainDataGenerator, TestDataGenerator
from lib import save_training, OptimizedRounder

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)
set_session(tf.Session(config=config))

###############################################################################################################################

model_name = "EfficientNetB5_224_regression"
cv = "4"

print('+++++++++++++++++++++++++++++++++++++++++++++++++++(  CV {} )+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'.format(cv))

os.mkdir(os.path.join("./cv_save/{}".format(model_name+"_cv"+cv)))

model = keras.models.load_model('./save/pretrain/'+model_name+"_pretrain.h5")
img_size = 224

# df_trainをさらに訓練データ（train_data）, 検証データ(val_data)に分ける
train_data = pd.read_csv('./CV_data/train_cv{}.csv'.format(cv))
val_data = pd.read_csv('./CV_data/valid_cv{}.csv'.format(cv))


train_generator = TrainDataGenerator(df=train_data, batch_size=16,
                                     augmentation=True, shuffle=True)
val_generator = TrainDataGenerator(df=val_data, batch_size=16)

early_stopping = EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5, verbose=1)

model.compile(loss="mse", optimizer=optimizers.Adam(lr=5e-4), metrics=["acc"])

history = model.fit_generator(train_generator, epochs=14, validation_data=val_generator,
                              steps_per_epoch=train_generator.__len__(),
                              callbacks=[reduce_lr], verbose=1)

plot_training(history, path=os.path.join("./cv_save/{}".format(model_name+"_cv"+cv)))

# modelの保存
model.save(os.path.join("./cv_save/{}".format(model_name+"_cv"+cv), model_name+"_cv"+cv+".h5"))


y_pred = []
y_true = []

for i, (img, label) in tqdm(enumerate(val_generator)):

    if i == val_generator.__len__():
        break

    pred_vec = model.predict(img)
    
    y_pred.extend(list(pred_vec))
    y_true.extend(list(label))
    

optR = OptimizedRounder()

print()
print('kappa score: ', cohen_kappa_score(y_true, optR.predict(y_pred, [0.5, 1.5, 2.5, 3.5]), weights="quadratic"))


# Optimize on validation data and evaluate again
optR = OptimizedRounder()
optR.fit(y_pred, y_true)
coefficients = optR.coefficients()
opt_val_predictions = optR.predict(y_pred, coefficients)
new_val_score = cohen_kappa_score(y_true, opt_val_predictions, weights="quadratic")
print("coefficients: ", coefficients)
print('new_val_score: ', new_val_score)


print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
