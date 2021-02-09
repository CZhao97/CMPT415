from __future__ import absolute_import, division, print_function, unicode_literals
import os
import json
import ember
import csv
import numpy as np
import lightgbm as lgb
from sklearn import preprocessing
# import tensorflow as tf
# from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.python.keras.utils import to_categorical
from keras.utils import to_categorical

data_dir_jan = 'ember_2017_2_01'

data_dir_feb = 'ember_2017_2_02'

data_dir_mar = 'ember_2017_2_03'


X_train_jan, y_train_jan = ember.read_vectorized_features(data_dir_jan, subset = 'train', feature_version=2)

X_train_feb, y_train_feb = ember.read_vectorized_features(data_dir_feb, subset = 'train', feature_version=2)

X_test, y_test = ember.read_vectorized_features(data_dir_mar,  subset = 'train', feature_version=2)

X_train = np.concatenate((X_train_jan,X_train_feb), axis = 0)

y_train = np.concatenate((y_train_jan,y_train_feb), axis = 0)

train_rows = (y_train != -1)

test_rows = (y_test != -1)

X_train = preprocessing.normalize(X_train[train_rows], norm='l2', axis = 0)

y_train = y_train[train_rows]

X_test = preprocessing.normalize(X_test[test_rows], norm='l2', axis = 0)

y_test = y_test[test_rows]


# train the model

n_inputs = 2381

print("Building neural network")
model = Sequential()
model.add(Dense(70, input_shape=(n_inputs,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(70, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

print("Compiling neural network")
# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

checkpath = os.path.join(os.path.dirname(os.path.abspath('')), "checkpoints")
os.makedirs(checkpath, exist_ok=True)
checkpath = os.path.join(checkpath, 'model-epoch{epoch:03d}-acc{val_acc:03f}.h5')

# stopper = EarlyStopping(monitor = 'val_acc', min_delta=0.0001, patience = 5, mode = 'auto')

# saver = ModelCheckpoint(checkpath, save_best_only=True, verbose=1, monitor='val_loss', mode='min')

print("Training neural network...")
# train the model
#! error with validation_data shape..
fitted_model = model.fit(X_train, y_train,
          epochs=3,
          verbose=2, 
          validation_data=(X_test, y_test)
         )


y_binary = to_categorical(y_test)
print(y_binary.shape)

# params = {
#         "boosting": "gbdt",
#         "objective": "binary",
#         "num_iterations": 1000,
#         "learning_rate": 0.05,
#         "num_leaves": 2048,
#         "max_depth": 15,
#         "min_data_in_leaf": 50,
#         "feature_fraction": 0.5,
#         "verbose" : -1，
#         # "application": "binary"
# }


# print("training lightGBM model")
# lgbm_dataset = lgb.Dataset(X_train, y_train)
# lgbm_model = lgb.train(params, lgbm_dataset)
# os.makedirs('lightgbm_models', exist_ok=True)
# lgbm_model.save_model(os.path.join('lightgbm_models', "model.txt"))