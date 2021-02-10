# import ff
import evaluate
import superficial_data

import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np

from swag import models, utils, losses
from swag.posteriors import SWAG

def main():

	def schedule(epoch):

		"""
		Function to get the learning rate based on current epoch when learning rate is not defined in advance

		Parameters:
		----------
		* epoch: current epoch

		Result:
		* assumed learning rate
		"""

		t = (epoch) / (args.swa_start if args.swa else args.epochs)
		lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
		if t <= 0.5:
		    factor = 1.0
		elif t <= 0.9:
		    factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
		else:
		    factor = lr_ratio
		return args.lr_init * factor

	# Extract data from super facial dataset and load into torch loaders

	train_dataset = superficial_data.SuperficialData_all("/mnt/c/Users/czhao/MalEvo/data/only_superficials/", train=True) # 51 x 1
    
	test_dataset = superficial_data.SuperficialData_all("/mnt/c/Users/czhao/MalEvo/data/only_superficials/", train=False) # 51 x 1


	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

	print(superficial_data.SuperficialData_all("/mnt/c/Users/czhao/MalEvo/data/only_superficials/", train=True))

	print(train_loader)


	# set all required parameters in advance
	parser = argparse.ArgumentParser(description="SGD/SWA training")

	# path to save checkpoint for predefined epoch
	parser.add_argument(
	    "--dir",
	    type=str,
	    default="models/checkpoint",
	    help="training directory (default: None)",
	)

	# path to access the superfacial data
	parser.add_argument(
	    "--data_path",
	    type=str,
	    default="/mnt/c/Users/czhao/MalEvo/data/only_superficials/",
	    metavar="PATH",
	    help="path to datasets location (default: None)",
	)
 

	# batch size defined with default value 32
	parser.add_argument(
	    "--batch_size",
	    type=int,
	    default=32,
	    metavar="N",
	    help="input batch size (default: 128)",
	)


	# set default model to deploy on SWAG
	parser.add_argument(
	    "--model",
	    type=str,
	    default="Feedforward_model",
	    metavar="MODEL",
	    help="model name (default: None)",
	)

	# define the default number of epoches
	parser.add_argument(
	    "--epochs",
	    type=int,
	    default=100,
	    metavar="N",
	    help="number of epochs to train (default: 200)",
	)

	# save the training weight baesd on defined frequency
	parser.add_argument(
	    "--save_freq",
	    type=int,
	    default=20,
	    metavar="N",
	    help="save frequency (default: 25)",
	)

	# moment update frequency to evaluate on test dataset
	parser.add_argument(
	    "--eval_freq",
	    type=int,
	    default=5,
	    metavar="N",
	    help="evaluation frequency (default: 5)",
	)

	# initialization of learning rate
	parser.add_argument(
	    "--lr_init",
	    type=float,
	    default=0.01,
	    metavar="LR",
	    help="initial learning rate (default: 0.01)",
	)

	# SGD momentum
	parser.add_argument(
	    "--momentum",
	    type=float,
	    default=0.9,
	    metavar="M",
	    help="SGD momentum (default: 0.9)",
	)

	# weight decay
	parser.add_argument(
	    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
	)


	# True to use SWAG and False to use SGD
	parser.add_argument("--swa", action="store_true", default=True, help="swa usage flag (default: off)")

	# SWA start epoch number
	parser.add_argument(
	    "--swa_start",
	    type=float,
	    default=50,
	    metavar="N",
	    help="SWA start epoch number (default: 50)",
	)

	# Learning rate of SWAG
	parser.add_argument(
	    "--swa_lr", type=float, default=0.05, metavar="LR", help="SWA LR (default: 0.02)"
	)

	# SWA model collection frequency/cycle length in epochs ('c' in SWAG algorithm)
	parser.add_argument(
	    "--swa_c_epochs",
	    type=int,
	    default=1,
	    metavar="N",
	    help="SWA model collection frequency/cycle length in epochs (default: 1)",
	)

	# Flag to decide if sample covariance needs to be saved
	parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")

	# maximum number of SWAG models to save
	parser.add_argument(
	    "--max_num_models",
	    type=int,
	    default=20,
	    help="maximum number of SWAG models to save",
	)

	# Load previous aquired checkpoint
	parser.add_argument(
	    "--swa_resume",
	    type=str,
	    default=None,
	    metavar="CKPT",
	    help="checkpoint to restor SWA from (default: None)",
	)

	# loss function for model training
	parser.add_argument(
	    "--loss",
	    type=str,
	    default="CE",
	    help="loss to use for training model (default: Cross-entropy)",
	)

	# set seed in random
	parser.add_argument(
	    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
	)

	# Flag to define if learning rate is porovided
	parser.add_argument("--no_schedule", action="store_true", help="store schedule")

	args = parser.parse_args()

	args.device = None

	use_cuda = torch.cuda.is_available()

	if use_cuda:
	    args.device = torch.device("cuda")
	else:
	    args.device = torch.device("cpu")


	# load training data and test data from superfacial data
	train_loader = torch.utils.data.DataLoader(superficial_data.SuperficialData_all(args.data_path, train=True), batch_size=32, shuffle=True)

	print(superficial_data.SuperficialData_all(args.data_path, train=True))

	print(train_loader)

	test_loader = torch.utils.data.DataLoader(superficial_data.SuperficialData_all(args.data_path, train=False), batch_size=32, shuffle=True)

	# create Directory if it does not exist
	print("Preparing directory %s" % args.dir)
	os.makedirs(args.dir, exist_ok=True)
	with open(os.path.join(args.dir, "command.sh"), "w") as f:
	    f.write(" ".join(sys.argv))
	    f.write("\n")

	torch.backends.cudnn.benchmark = True
	torch.manual_seed(args.seed)

	if use_cuda:
		torch.cuda.manual_seed(args.seed)

	# get all parameters of model and store in model configuration variable
	print("Using model %s" % args.model)
	model_cfg = getattr(models, args.model)

	num_classes = 2

	# dimension of training dataset
	n_dim = train_dataset.__dimension__()

	# input paramaterrs to our model
	print("Preparing model")
	print(*model_cfg.args)
	model = model_cfg.base( n_classes=num_classes, n_dim=n_dim, *model_cfg.args,  **model_cfg.kwargs)
	model.to(args.device)


	if args.cov_mat:
	    args.no_cov_mat = False
	else:
	    args.no_cov_mat = True

	# Set SWAG if swa flag is defined as TRUE, else using SGD
	if args.swa:
	    print("SWAG training")
	    swag_model = SWAG(
	        base = model_cfg.base,
	        no_cov_mat=args.no_cov_mat,
	        max_num_models=args.max_num_models,
	        num_classes=num_classes,
	        n_dim=n_dim,
	        *model_cfg.args,
	        **model_cfg.kwargs
	    )
	    swag_model.to(args.device)
	else:
	    print("SGD training")

	# set loss function
	if args.loss == "CE":
	    criterion = losses.cross_entropy
	elif args.loss == "adv_CE":
	    criterion = losses.adversarial_cross_entropy

	# set optimizer with all parameters from model and momentum, learning rate and weght decay
	optimizer = torch.optim.SGD(
	    model.parameters(), 
	    lr=args.lr_init, 
	    momentum=args.momentum, 
	    weight_decay=args.wd
	)

	# initial status
	start_epoch = 0

	# defined if checkpoint needed to be uploaded
	if args.swa and args.swa_resume is not None:
	    checkpoint = torch.load(args.swa_resume)
	    swag_model = SWAG(
	        base = model_cfg.base,
	        no_cov_mat=args.no_cov_mat,
	        max_num_models=args.max_num_models,
	        num_classes=num_classes,
	        loading=True,
	        n_dim=n_dim,
	        *model_cfg.args,
	        **model_cfg.kwargs
	    )
	    print("*"*100)
	    swag_model.to(args.device)
	    swag_model.load_state_dict(checkpoint["state_dict"])

	columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time"]
	if args.swa:
	    columns = columns[:-2] + ["swa_te_loss", "swa_te_acc"] + columns[-2:]
	    swag_res = {"loss": None, "accuracy": None}

	# save the initial status as the first checkpoint
	utils.save_checkpoint(
	    args.dir,
	    start_epoch,
	    state_dict=model.state_dict(),
	    optimizer=optimizer.state_dict(),
	)

	sgd_ens_preds = None
	sgd_targets = None
	n_ensembled = 0.0

	# start training our model in epoches
	for epoch in range(start_epoch, args.epochs):
	    time_ep = time.time()

	    # update learning rate if learning rate is not given
	    if not args.no_schedule:
	        lr = schedule(epoch)
	        utils.adjust_learning_rate(optimizer, lr)
	    else:
	        lr = args.lr_init

	    train_res = utils.train_epoch(train_loader, model, criterion, optimizer, cuda=use_cuda)


	    # evaluate on test dataset based on predefined evaluation frequency
	    if (
	        epoch == 0
	        or epoch % args.eval_freq == args.eval_freq - 1
	        or epoch == args.epochs - 1
	    ):

	        test_res = utils.eval(test_loader, model, criterion, cuda=use_cuda)
	    else:
	        test_res = {"loss": None, "accuracy": None}

	    # update our model using SWAG at the first iteration and every 'c' epoches
	    if (
	        args.swa
	        and epoch >= args.swa_start
	        and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0
	    ):

	        sgd_res = utils.predict(test_loader, model)

	        sgd_preds = sgd_res["predictions"]
	        sgd_targets = sgd_res["targets"]
	        print("updating sgd_ens")

	        if sgd_ens_preds is None:
	            sgd_ens_preds = sgd_preds.copy()
	        else:
	            # TODO: rewrite in a numerically stable way
	            sgd_ens_preds = sgd_ens_preds * n_ensembled / (
	                n_ensembled + 1
	            ) + sgd_preds / (n_ensembled + 1)
	        n_ensembled += 1

	        # update SWAG when model updated
	        swag_model.collect_model(model)

	        # evaluate SWAG for the first, last and every eval_freq epoches
	        if (
	            epoch == 0
	            or epoch % args.eval_freq == args.eval_freq - 1
	            or epoch == args.epochs - 1
	        ):

	            swag_model.sample(0.0)

	            # Performs 1 epochs to estimate buffers average using train dataset

	            utils.bn_update(train_loader, swag_model)
	            
	            swag_res = utils.eval(test_loader, swag_model, criterion)
	        else:
	            swag_res = {"loss": None, "accuracy": None}

	    # save checkpoint for every save_freq iterations
	    if (epoch + 1) % args.save_freq == 0:
	        utils.save_checkpoint(
	            args.dir,
	            epoch + 1,
	            state_dict=model.state_dict(),
	            optimizer=optimizer.state_dict(),
	        )
	        if args.swa:
	            utils.save_checkpoint(
	                args.dir, epoch + 1, name="swag", state_dict=swag_model.state_dict()
	            )

	    time_ep = time.time() - time_ep
	    
	        
	    values = [
	        epoch + 1,
	        lr,
	        train_res["loss"],
	        train_res["accuracy"],
	        test_res["loss"],
	        test_res["accuracy"],
	        time_ep,
	    ]

	    # print result
	    if args.swa:
	        values = values[:-2] + [swag_res["loss"], swag_res["accuracy"]] + values[-2:]
	    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
	    if epoch % 40 == 0:
	        table = table.split("\n")
	        table = "\n".join([table[1]] + table)
	    else:
	        table = table.split("\n")[2]

	    print(table)


	# save the last checkpoint after all epoch
	if args.epochs % args.save_freq != 0:
	    utils.save_checkpoint(
	        args.dir,
	        args.epochs,
	        state_dict=model.state_dict(),
	        optimizer=optimizer.state_dict(),
	    )
	    if args.swa and args.epochs > args.swa_start:
	        utils.save_checkpoint(
	            args.dir, args.epochs, name="swag", state_dict=swag_model.state_dict()
	        )

	if args.swa:
	    np.savez(
	        os.path.join(args.dir, "sgd_ens_preds.npz"),
	        predictions=sgd_ens_preds,
	        targets=sgd_targets,
	    )


	# print(utils.predict(test_loader,swag_model))


if __name__ == "__main__":
    main()
