"""
Includes functions for evaluation.

Author: Arash Khoeini
Email: arash.khoeini@gmail.com

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.calibration import calibration_curve
import os

def draw_roc(folds_result, name, plot_path):
    """
    Function to draw ROC plots for folds and calculate the TPR at FPR=0.01

    Parameters:
    ----------
    * folds_result : list
         list of fold results. Each element in this array is as [target, predicted] where
         target and predicted are arrays themselves.
    * name : str
        name of the file to be saved.
    """
    
    cm_list = []
    tpr_list = []
    auc_list = []
    cm_list = []
    mean_fpr = np.linspace(0,1,100)
    for i, fold_result in enumerate(folds_result):


        predicted = fold_result[1] 

        fpr, tpr, _ = roc_curve(fold_result[0], predicted)

        tpr_list.append(interp(mean_fpr, fpr, tpr))
        tpr_list[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        auc_list.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
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

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    plt.savefig(os.path.join(plot_path , f'{name}.png'))
    plt.close()


def draw_reliability_diagram(models_result, name, n_bins=5):
    """ A method to draw reliability diagrams for multiple models


    Parameters:
    ----------
    * models_result: dict
        a dictionary where key is model name and value is a tuple of the form (predicted, target)
    * name: str
        name of the file to be saved
    * n_bins: int optional
        number of bins to draw reliability diagram

    """
    for key in models_result:
        pred = models_result[key][0]
        target = models_result[key][1]

        fraction_of_positives, mean_predicted_value = calibration_curve(target, pred, n_bins=n_bins)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=key)
        plt.ylabel("Fraction of positives")
        plt.ylim([-0.05, 1.05])
        plt.xlim([-0.05, 1.05])
        plt.legend(loc="lower right")
        plt.title('Calibration plots  (reliability curve)')

        if not os.path.exists('plots'):
            os.makedirs('plots')

        plt.savefig(os.path.join('plots' , f'{name}.png'))
        plt.close()
