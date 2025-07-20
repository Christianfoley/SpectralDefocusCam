import torch.optim as optim
import torch.nn as nn
import torch
import copy


def get_optimizer(model, name, kwparams):
    """
    Initializes and returns an optimizer based upon the given params
    """
    if name == "sgd":
        optimizer = optim.SGD
    elif name == "adam":
        optimizer = optim.Adam
    elif name == "adamw":
        optimizer = optim.AdamW
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
    elif name == "fista_net":
        lfn = fista_net_loss
    elif name == "ssim":
        raise NotImplementedError(
            "See https://github.com/VainF/pytorch-msssim/tree/master"
        )
    else:
        raise NotImplementedError()

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


class fista_net_loss(nn.Module):
    def __init__(self, lambda_sym=0.01, lambda_st=0.001):
        """
        Composite MSE, L1, Sparsity and Symmetry loss as explained in:

        Forward Parameters
        ----------
        outputs : list
            (prediction, sym loss list, sparsity loss list)
        y : torch.Tensor
            ground truth
        """
        super(fista_net_loss, self).__init__()
        self.lambda_sym = lambda_sym
        self.lambda_st = lambda_st

    def forward(self, outputs, y):
        pred, loss_layers_sym, loss_layers_st = outputs

        if len(pred.shape) > 4:
            pred = torch.squeeze(pred, 1)

        # Compute loss, data consistency and regularizer constraints
        loss_discrepancy = nn.MSELoss()(pred, y) + 0.1 * nn.L1Loss()(pred, y)
        loss_constraint = 0
        for k, _ in enumerate(loss_layers_sym, 0):
            loss_constraint += torch.mean(torch.pow(loss_layers_sym[k], 2))

        sparsity_constraint = 0
        for k, _ in enumerate(loss_layers_st, 0):
            sparsity_constraint += torch.mean(torch.abs(loss_layers_st[k]))

        # sym and st weights are extremely small, as norms have high base vals
        loss = (
            loss_discrepancy
            + self.lambda_sym * loss_constraint
            + self.lambda_st * sparsity_constraint
        )
        return loss.to(torch.float32)
