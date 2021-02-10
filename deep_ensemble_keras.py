#!/usr/bin/env python
# coding: utf-8

# In[17]:


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
from keras.utils import to_categorical
from keras.models import load_model
from tensorflow import keras



# In[25]:


def extract_data(nomalization = True):
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
    
    if nomalization:
        X_train = preprocessing.normalize(X_train[train_rows], norm='l2', axis = 0)
    else:
        X_train = X_train[train_rows]
        
    y_train = y_train[train_rows]
    
    if nomalization:
        X_test = preprocessing.normalize(X_test[test_rows], norm='l2', axis = 0)
    else:
        X_test = X_test[test_rows]
        
    y_test = y_test[test_rows]

    return X_train, y_train, X_test, y_test

def create_model(dim_input = 2381, dim_output = 2, lr = 0.01):
    print("Building neural network")
    model = Sequential()
    model.add(Dense(1024, input_shape=(dim_input,), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(dim_output, activation='softmax'))

    print("Compiling neural network")
    # compile the model
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def get_model_trained(X_train, y_train, X_test, y_test, model, epochs = 10):
#     n_inputs = 2381

    checkpath = os.path.join(os.path.dirname(os.path.abspath('')), "checkpoints")
    os.makedirs(checkpath, exist_ok=True)
    checkpath = os.path.join(checkpath, 'model-epoch{epoch:03d}-acc{val_acc:03f}.h5')

    stopper = EarlyStopping(monitor = 'val_acc', min_delta=0.0001, patience = 5, mode = 'auto')

    saver = ModelCheckpoint(checkpath, save_best_only=True, verbose=1, monitor='val_loss', mode='min')

    print("Training neural network...")
    # train the model
    #! error with validation_data shape..
    model.fit(X_train, y_train,
              epochs=3,
              verbose=2, 
              callbacks=[stopper],
    #           validation_data=(X_test, y_test)
             )
    return model

def train_ensemble(X_train, y_train, X_test, y_test, num_model = 5):
    
    for i in range(5):
        model = create_model(2381, 2, 0.01)
        model = get_model_trained(X_train, y_train, X_test, y_test, model, 3)
        model_json = model.to_json()
        if not os.path.exists("models_keras"):
            os.makedirs("models_keras", exist_ok=True)
#         with open("models_keras/model_{}.json".format(i), "w") as json_file:
#             json_file.write(model_json)
#             json_file.close()
        # model_yaml = model.to_yaml()
        # with open("models_keras/model_{}.yaml".format(i), "w") as yaml_file:
        #     yaml_file.write(model_yaml)
        model.save_weights("models_keras/model_{}.h5".format(i))
        
def accuracy(y_hat, y):
    pred = np.argmax(y_hat, 1)
    return (pred == y).mean()


# In[26]:


X_train, y_train, X_test, y_test = extract_data()
train_ensemble(X_train, y_train, X_test, y_test)


# In[27]:


model = create_model()

# load weights into new model
model.load_weights("models_keras/model_0.h5")


# In[28]:


y_binary = to_categorical(y_test)
print(y_binary.shape)


# In[29]:


accuracy(model.predict(X_test),y_test)


# In[ ]:




