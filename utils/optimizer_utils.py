import torch.optim as optim
import torch.nn as nn
import copy


def get_optimizer(model, name, kwparams):
    """
    Initializes and returns an optimizer based upon the given params
    """
    if name == "sgd":
        optimizer = optim.SGD
    elif name == "adam":
        optimizer = optim.Adam
    elif name == "adagrad":
        optimizer = optim.Adagrad
    elif name == "rmsprop":
        optimizer = optim.RMSprop

    args = copy.deepcopy(kwparams)
    args["params"] = model.parameters()
    optimizer = optimizer(**args)
    return optimizer


def get_lr_scheduler(optimizer, name, kwparams):
    """
    Initializes and returns an lr scheduler based upon the given params
    """
    if name == "warm_cosine_anneal":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts
    elif name == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR
    elif name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau
    args = copy.deepcopy(kwparams)
    args["optimizer"] = optimizer
    scheduler = scheduler(**args)
    return scheduler


def get_loss_function(name, kwparams=None):
    """
    Initializes and returns a loss funciton based upon the given params
    """
    if name == "mse" or name == "l2":
        lfn = nn.MSELoss
    elif name == "mae" or name == "l1":
        lfn = nn.L1Loss
    elif name == "cossim":
        lfn = nn.CosineEmbeddingLoss
    elif name == "ssim":
        raise NotImplementedError(
            "See https://github.com/VainF/pytorch-msssim/tree/master"
        )

    if kwparams is None:
        kwparams = {}
    lfn = lfn(**kwparams)
    return lfn


class NoamOpt:
    """
    Warmup annealing cosine learning rate scheduler from Attention Is All You Need:
        code: https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer
        paper: https://arxiv.org/abs/1706.03762
    """

    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


def get_lr(optimizer):
    """
    Get current learning rate from a pytorch optimizer with variable lr
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
