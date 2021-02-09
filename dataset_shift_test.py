import numpy as np
import torch
import random
from torch import *
from torch_two_sample import *
from scipy.stats import ks_2samp
from scipy.spatial import distance

def ks_test(self, X_tr, X_te):
	p_vals = []

	for i in range(X_tr.shape[1]):
		feature_tr = X_tr[:, i]
		feature_te = X_te[:, i]

		t_val, p_val = None, None

		if self.ot == OnedimensionalTest.KS:

			# Compute KS statistic and p-value
			t_val, p_val = ks_2samp(feature_tr, feature_te)
		elif self.ot == OnedimensionalTest.AD:
			t_val, _, p_val = anderson_ksamp([feature_tr.tolist(), feature_te.tolist()])

		p_vals.append(p_val)


	p_vals = np.array(p_vals)
	p_val = min(np.min(p_vals), 1.0)

	return p_val, p_vals

def MMA_test(self, X_tr, X_te):

	# torch_two_sample somehow wants the inputs to be explicitly casted to float 32.
	X_tr = X_tr.astype(np.float32)
	X_te = X_te.astype(np.float32)

	p_val = None

	# We provide a couple of different tests, although we only report results for MMD in the paper.
	if self.mt == MultidimensionalTest.MMD:
		mmd_test = MMDStatistic(len(X_tr), len(X_te))

		# As per the original MMD paper, the median distance between all points in the aggregate sample from both
		# distributions is a good heuristic for the kernel bandwidth, which is why compute this distance here.
		if len(X_tr.shape) == 1:
			X_tr = X_tr.reshape((len(X_tr),1))
			X_te = X_te.reshape((len(X_te),1))
			all_dist = distance.cdist(X_tr, X_te, 'euclidean')
		else:
			all_dist = distance.cdist(X_tr, X_te, 'euclidean')
		median_dist = np.median(all_dist)

		# Calculate MMD.
		t_val, matrix = mmd_test(torch.autograd.Variable(torch.tensor(X_tr)),
									 torch.autograd.Variable(torch.tensor(X_te)),
									 alphas=[1/median_dist], ret_matrix=True)
		p_val = mmd_test.pval(matrix)
	elif self.mt == MultidimensionalTest.Energy:
		energy_test = EnergyStatistic(len(X_tr), len(X_te))
		t_val, matrix = energy_test(torch.autograd.Variable(torch.tensor(X_tr)),
									torch.autograd.Variable(torch.tensor(X_te)),
									ret_matrix=True)
		p_val = energy_test.pval(matrix)
	elif self.mt == MultidimensionalTest.FR:
		fr_test = FRStatistic(len(X_tr), len(X_te))
		t_val, matrix = fr_test(torch.autograd.Variable(torch.tensor(X_tr)),
									torch.autograd.Variable(torch.tensor(X_te)),
									norm=2, ret_matrix=True)
		p_val = fr_test.pval(matrix)
	elif self.mt == MultidimensionalTest.KNN:
		knn_test = KNNStatistic(len(X_tr), len(X_te), 20)
		t_val, matrix = knn_test(torch.autograd.Variable(torch.tensor(X_tr)),
									 torch.autograd.Variable(torch.tensor(X_te)),
									 norm=2, ret_matrix=True)
		p_val = knn_test.pval(matrix)
	    
	return p_val, np.array([])



# https://github.com/steverab/failing-loudly/blob/master/shift_tester.py