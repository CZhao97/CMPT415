import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

def cross_entropy(model, input, target):
    """
    standard cross-entropy loss function

    Parameters:
    ----------
    * model: input model
    * input: input value (list)
    * target: target value (list)


    Result:
    * loss score and list of predicted output
    """


    output = model(input.float())

    loss = F.cross_entropy(output.float(), target)

    return loss, output

def adversarial_cross_entropy(
    model, input, target, lossfn=F.cross_entropy, epsilon=0.01
):
    # loss function based on algorithm 1 of "simple and scalable uncertainty estimation using
    # deep ensembles," lakshminaraynan, pritzel, and blundell, nips 2017,
    # https://arxiv.org/pdf/1612.01474.pdf
    # note: the small difference bw this paper is that here the loss is only backpropped
    # through the adversarial loss rather than both due to memory constraints on preresnets
    # we can change back if we want to restrict ourselves to VGG-like networks (where it's fine).

    # scale epsilon by min and max (should be [0,1] for all experiments)
    # see algorithm 1 of paper
    scaled_epsilon = epsilon * (input.max() - input.min())

    # force inputs to require gradient
    input.requires_grad = True

    # standard forwards pass
    output = model(input)
    loss = lossfn(output, target)

    # now compute gradients wrt input
    loss.backward(retain_graph=True)

    # now compute sign of gradients
    inputs_grad = torch.sign(input.grad)

    # perturb inputs and use clamped output
    inputs_perturbed = torch.clamp(
        input + scaled_epsilon * inputs_grad, 0.0, 1.0
    ).detach()
    # inputs_perturbed.requires_grad = False

    input.grad.zero_()
    # model.zero_grad()

    outputs_perturbed = model(inputs_perturbed)

    # compute adversarial version of loss
    adv_loss = lossfn(outputs_perturbed, target)

    # return mean of loss for reasonable scalings
    return (loss + adv_loss) / 2.0, output


