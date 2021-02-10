"""
    implementation of SWAG
"""

import torch
import numpy as np
import itertools
from torch.distributions.normal import Normal
import copy

import gpytorch
from gpytorch.lazy import RootLazyTensor, DiagLazyTensor, AddedDiagLazyTensor
from gpytorch.distributions import MultivariateNormal

from ..utils import flatten, unflatten_like


def swag_parameters(module, params, no_cov_mat=True):

    """
    initialize mean and square of mean for every not none module and save in buffer

    Parameters:
    ----------
    * module: current neural network weights
    * params: list of module
    * no_cov_mat: flag of covariance matrix
    """

    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            continue
        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer("%s_mean" % name, data.new(data.size()).zero_())
        module.register_buffer("%s_sq_mean" % name, data.new(data.size()).zero_())

        if no_cov_mat is False:
            module.register_buffer(
                "%s_cov_mat_sqrt" % name, data.new_empty((0, data.numel())).zero_()
            )

        params.append((module, name))


class SWAG(torch.nn.Module):
    def __init__(
        self, base, no_cov_mat=True, max_num_models=0, var_clamp=1e-30, num_classes = 0,n_dim = 0,*args, **kwargs
    ):
        super(SWAG, self).__init__()

        self.register_buffer("n_models", torch.zeros([1], dtype=torch.long))
        self.params = list()

        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models

        self.var_clamp = var_clamp

        self.n_dim = n_dim
        self.num_classes = num_classes

        self.base = base(n_dim,num_classes,*args, **kwargs)
        self.base.apply(
            lambda module: swag_parameters(
                module=module, params=self.params, no_cov_mat=self.no_cov_mat
            )
        )

    def forward(self, *args, **kwargs):

        """
        Forward function

        Parameters:
        ----------
        * args: number of parameters
        * kwargs: weights

        Result:
        * call 'base' (model) function to move forward
        """

        return self.base(*args, **kwargs)

    def sample(self, scale=1.0, cov=False, seed=None, block=False, fullrank=True):

        """
        get the sample of normal distribution from mean and square root

        Parameters:
        ----------
        * scale: scale value
        * cov: covatiance matrix
        * seed: set seed to manually set random values
        * block: flag to determine sample from full rank or from blockwise
        * fullrank: used in blockwise sample to determine if covariance matrix sample will be used

        """

        if seed is not None:
            torch.manual_seed(seed)

        if not block:
            self.sample_fullrank(scale, cov)
        else:
            self.sample_blockwise(scale, cov, fullrank)


    def sample_fullrank(self, scale, cov):

        """
        get the sample of gaussian distribution based on all means and square roots

        Parameters:
        ----------
        * scale: scale value
        * cov: covatiance matrix

        Result:
        * get the Gaussion distribution from all ranks
        """

        scale_sqrt = scale ** 0.5

        mean_list = []
        sq_mean_list = []

        if cov:
            cov_mat_sqrt_list = []

        for (module, name) in self.params:
            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)

            if cov:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                cov_mat_sqrt_list.append(cov_mat_sqrt.cpu())

            mean_list.append(mean.cpu())
            sq_mean_list.append(sq_mean.cpu())

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty(
                    (cov_mat_sqrt.size(0),), requires_grad=False
                ).normal_()
            )
            cov_sample /= (self.max_num_models - 1) ** 0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)

        for (module, name), sample in zip(self.params, samples_list):
            module.__setattr__(name, sample)
            # module.__setattr__(name, sample.cuda())

    def sample_blockwise(self, scale, cov, fullrank):

        """
        get the sample of normal distribution from each mean and square root

        Parameters:
        ----------
        * scale: scale value
        * cov: covatiance matrix
        * fullrank: flag to determine if the previous covariance matrix square root will be used

        Result:
        * get the Gaussion distribution from all ranks
        """

        for module, name in self.params:
            mean = module.__getattr__("%s_mean" % name)

            sq_mean = module.__getattr__("%s_sq_mean" % name)
            eps = torch.randn_like(mean)

            var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)

            scaled_diag_sample = scale * torch.sqrt(var) * eps

            if cov is True:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)
                eps = cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0), 1)).normal_()
                cov_sample = (
                    scale / ((self.max_num_models - 1) ** 0.5)
                ) * cov_mat_sqrt.t().matmul(eps).view_as(mean)

                if fullrank:
                    w = mean + scaled_diag_sample + cov_sample
                else:
                    w = mean + scaled_diag_sample

            else:
                w = mean + scaled_diag_sample

            module.__setattr__(name, w)

    def collect_model(self, base_model):

        """
        update the attributes and store in the predefined buffer

        Parameters:
        ----------
        * base_model: collect all parameters from base_model and deploy to SWAG class

        """

        for (module, name), base_param in zip(self.params, base_model.parameters()):

            mean = module.__getattr__("%s_mean" % name)
            sq_mean = module.__getattr__("%s_sq_mean" % name)


            # first moment
            mean = mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data / (self.n_models.item() + 1.0)

            # second moment
            sq_mean = sq_mean * self.n_models.item() / (
                self.n_models.item() + 1.0
            ) + base_param.data ** 2 / (self.n_models.item() + 1.0)

            # square root of covariance matrix
            if self.no_cov_mat is False:
                cov_mat_sqrt = module.__getattr__("%s_cov_mat_sqrt" % name)

                # block covariance matrices, store deviation from current mean
                dev = (base_param.data - mean).view(-1, 1)
                cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev.view(-1, 1).t()), dim=0)

                # remove first column if we have stored too many models
                if (self.n_models.item() + 1) > self.max_num_models:
                    cov_mat_sqrt = cov_mat_sqrt[1:, :]
                module.__setattr__("%s_cov_mat_sqrt" % name, cov_mat_sqrt)

            module.__setattr__("%s_mean" % name, mean)
            module.__setattr__("%s_sq_mean" % name, sq_mean)
        self.n_models.add_(1)

    def load_state_dict(self, state_dict, strict=True):

        """
        load weights from dictionary (checkpoints)

        Parameters:
        ----------
        * state_dict: state dictionary of checkpoints

        """

        if not self.no_cov_mat:
            n_models = state_dict["n_models"].item()
            rank = min(n_models, self.max_num_models)
            for module, name in self.params:
                mean = module.__getattr__("%s_mean" % name)
                module.__setattr__(
                    "%s_cov_mat_sqrt" % name,
                    mean.new_empty((rank, mean.numel())).zero_(),
                )
        super(SWAG, self).load_state_dict(state_dict, strict)