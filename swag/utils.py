import itertools
import torch
import os
import copy
from datetime import datetime
import math
import numpy as np
import tqdm
import torch.nn.functional as F



def flatten(lst):

    """
    change the dimension and memory allocation of list of data

    Parameters:
    ----------
    * lst: list of data

    Result:
    * return list in one dimension
    """

    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i : i + n].view(tensor.shape))
        i += n
    return outList


def adjust_learning_rate(optimizer, lr):

    """
    update learning rate in optimizer (when learning rate is not predefined)

    Parameters:
    ----------
    * optimizer: optimizer
    * lr: learning rate

    Result:
    * learning rate
    """

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_checkpoint(dir, epoch, name="checkpoint", **kwargs):

    """
    save the current checkpoint to local

    Parameters:
    ----------
    * dir: path to store checkpoint
    * epoch: current epoch
    * **kwargs: weights

    Result:
    * save checkpoint named by the combination of epoch and name in dir 
    """

    state = {"epoch": epoch}
    state.update(kwargs)
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)


def train_epoch(
    train_dataset,
    model,
    criterion,
    optimizer,
    num_batch=32,
    cuda=False,
    regression=False,
    verbose=False
):

    """
    function to train our model based on diverse epoch

    Parameters:
    ----------
    * train_dataset: training dataset
    * model: model
    * criterion: loss function
    * optimizer: optimizer
    * num_batch: number of batch
    * cuda: flag of cuda
    * **kwargs: weights

    Result:
    * return value of loss score and accuracy
    """
    loss_sum = 0.0
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0

    model.train()

    for i, (input, target) in enumerate(train_dataset):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)


        loss, output = criterion(model, input, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.data.item() * input.size(0)

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batch >= verb_stage + 1:
            print(
                "Stage %d/10. Loss: %12.4f. Acc: %6.2f"
                % (
                    verb_stage + 1,
                    loss_sum / num_objects_current,
                    correct / num_objects_current * 100.0,
                )
            )
            verb_stage += 1

    return {
        "loss": loss_sum / num_objects_current,
        "accuracy": None if regression else correct / num_objects_current * 100.0,
    }



def eval(test_dataset, model, criterion, cuda=False, regression=False, verbose=False):

    """
    evaluation function with return values loss and accuracy

    Parameters:
    ----------
    * test_dataset: testing dataset
    * model: model
    * criterion: loss function
    * num_batch: number of batch
    * cuda: flag of cuda
    * **kwargs: weights

    Result:
    * return value of loss score and accuracy
    """

    loss_sum = 0.0
    correct = 0.0
    num_objects_total = test_dataset.__len__()
    model.eval()

    with torch.no_grad():
        if verbose:
            loader = tqdm.tqdm(loader)
        for i, (input, target) in enumerate(test_dataset):
            if cuda:
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            loss, output = criterion(model, input, target)

            loss_sum += loss.item() * input.size(0)

            if not regression:
                pred = output.data.argmax(1, keepdim=True)

                correct += pred.eq(target.data.view_as(pred)).sum().item()

    return {
        "loss": loss_sum / num_objects_total,
        "accuracy": None if regression else correct / num_objects_total * 100.0,
    }



def predict(loader, model, verbose=False):

    """
    function to predict the result in format of 0 or 1

    Parameters:
    ----------
    * loader: torch data loader
    * model: model

    Result:
    * return dictionary of outputs and targets
    """

    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for input, target in loader:
            # input = input.cuda(non_blocking=True)
            output = model(input.float())

            batch_size = input.size(0)

            predictions.append(output.data.argmax(1, keepdim=True).cpu().numpy())

            targets.append(target.numpy())
            offset += batch_size

    return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}


def predict_possibility(loader, model, verbose=False):

    """
    function to predict the result in format of possibility of two class

    Parameters:
    ----------
    * loader: torch data loader
    * model: model

    Result:
    * return dictionary of outputs and targets in possiblity
    """

    predictions = list()
    targets = list()

    model.eval()

    if verbose:
        loader = tqdm.tqdm(loader)

    offset = 0
    with torch.no_grad():
        for input, target in loader:
            # input = input.cuda(non_blocking=True)
            output = model(input.float())

            batch_size = input.size(0)

            predictions.extend(output.data.cpu().numpy()[:,1].tolist())

            targets.append(target.numpy())
            offset += batch_size

    return {"predictions": predictions, "targets": np.concatenate(targets)}



def _check_bn(module, flag):

    # check if the module class is the subclass of BatchNorm

    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model):

    # calling _check_bn function to return if modules in model is subclass of BatchNorm

    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]




def bn_update(loader, model, verbose=False, subset=None, **kwargs):
    """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.

        :param loader: train dataset loader for buffers average estimation.
        :param model: model being update
        :return: None
    """
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    num_batches = len(loader)

    with torch.no_grad():
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
        if verbose:

            loader = tqdm.tqdm(loader, total=num_batches)
        for input, _ in loader:
            input = input.cuda(non_blocking=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            model(input_var, **kwargs)
            n += b

    model.apply(lambda module: _set_momenta(module, momenta))

