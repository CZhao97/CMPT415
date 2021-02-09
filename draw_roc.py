import tarfile
import numpy as np 
import ember
import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.calibration import calibration_curve


X_train_df_2018 = pd.read_csv('ember2018/X_train.csv')
y_train_df_2018 = pd.read_csv('ember2018/y_train.csv')
X_test_df_2018 = pd.read_csv('ember2018/X_test.csv')
y_test_df_2018 = pd.read_csv('ember2018/y_test.csv')

# X_train_df_2017 = pd.read_csv('ember_2017_2/X_train.csv')
# y_train_df_2017 = pd.read_csv('ember_2017_2/y_train.csv')
X_test_df_2017 = pd.read_csv('ember_2017_2/X_test.csv')
y_test_df_2017 = pd.read_csv('ember_2017_2/y_test.csv')




temp_df = pd.concat([X_train_df_2018, y_train_df_2018], axis=1, sort=False)


temp_df = temp_df[temp_df.iloc[:,-1:]!=-1]
new_y_train_2018 = temp_df[temp_df.columns[-2:]]
new_x_train_2018 = temp_df[temp_df.columns[:-2]]



D_train_2018 = xgb.DMatrix(new_x_train_2018.values, label=new_y_train_2018.values)
D_test_2018 = xgb.DMatrix(X_test_df_2018.values, label=y_test_df_2018.values)

D_test_2017 = xgb.DMatrix(X_test_df_2017.values, label=y_test_df_2017.values)



params = {
    "boosting": "gbdt",
    'objective': 'binary:logistic',
    "num_iterations": 1000,
    "learning_rate": 0.05,
    "num_leaves": 2048,
    "max_depth": 15,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.5
}
steps = 20




model = xgb.train(params, D_train_2018, steps)



preds_2018 = model.predict(D_test_2018)
best_preds_2018 = np.asarray([np.argmax(line) for line in preds_2018])

preds_2017 = model.predict(D_test_2017)
best_preds_2017 = np.asarray([np.argmax(line) for line in preds_2017])



cm_list = []
tpr_list = []
auc_list = []
cm_list = []
mean_fpr = np.linspace(0,1,100)


predicted_2018 = list(preds_2018)

fpr, tpr, _ = roc_curve(new_y_train_2018["0"].values, predicted_2018)

tpr_list.append(interp(mean_fpr, fpr, tpr))
tpr_list[-1][0] = 0.0
roc_auc = auc(fpr, tpr)
auc_list.append(roc_auc)
plt.plot(fpr, tpr, lw=1, alpha=0.3,
        label='ROC 2018 (AUC = %0.2f)' % (roc_auc))


predicted_2017 = list(preds_2017)

fpr, tpr, _ = roc_curve(new_y_train_2017["0"].values, predicted_2017)

tpr_list.append(interp(mean_fpr, fpr, tpr))
tpr_list[-1][0] = 0.0
roc_auc = auc(fpr, tpr)
auc_list.append(roc_auc)
plt.plot(fpr, tpr, lw=1, alpha=0.3,
        label='ROC 2018 (AUC = %0.2f)' % (roc_auc))


plt.plot([0,1] , [0,1], linestyle='--', lw=2, color='r', label='Chance', 
       alpha=0.8)
mean_tpr = np.mean(tpr_list, axis=0)
mean_tpr[-1] = 1.0
#print(f"TPR at FPR=0.01 is: {mean_tpr[1]:.3f}")
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(auc_list)
plt.plot(mean_fpr, mean_tpr, color='b', 
       label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
       lw=2, alpha=.8)
std_tpr = np.std(tpr_list, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
              label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f"TPR at FPR=0.01 is: {mean_tpr[1]:.3f}")
plt.legend(loc="lower right")

plt.show()



