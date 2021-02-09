import ember
import pandas as pd
import numpy as np
import torch

# ember.create_vectorized_features("new_ember_2017_2")
# ember.create_metadata("new_ember_2017_2")

# ember.create_vectorized_features("new_ember2018")
# ember.create_metadata("new_ember2018")

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

X_train = X_train[train_rows]

y_train = y_train[train_rows]

X_test = X_test[test_rows]

y_test = y_test[test_rows]


# pd.DataFrame(X_train).to_csv('test_online/X_train.csv', index=False)
# pd.DataFrame(y_train).to_csv('test_online/y_train.csv', index=False)
# pd.DataFrame(X_test).to_csv('test_online/X_test.csv', index=False)
# pd.DataFrame(y_test).to_csv('test_online/y_test.csv', index=False)

X_train, y_train, X_test, y_test = map(torch.tensor, (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)))

print(X_train.shape)

print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# ember.create_vectorized_features("ember2018")
# lgbm_model = ember.train_model("ember2018")

# print(df)
# print(metadata_dataframe)