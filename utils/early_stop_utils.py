import numpy as np
import torch
import os


class EarlyStopping:
    def __init__(
        self,
        path,
        patience=7,
        verbose=False,
        delta=0,
        trace_func=print,
        save_model=True,
    ):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Adapted from:
            https://github.com/Bjarten/early-stopping-pytorch

        :param int patience: How long to wait after last time validation loss improved.
                            Default: 7
        :param bool verbose: If True, prints a message for each validation loss improvement.
                            Default: False
        :param float delta: Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        :param str path: Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        :param funct trace_func: trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.path = path
        self.save_model = save_model

    def __call__(self, val_loss, model, epoch):
        """
        Determine whether stopping is necessary this epoch.
        Should be called every epoch to enforce early stopping.

        :param float val_loss: avg loss from validation dataset
        :param nn.Module model: model from which to save early
        :param int epoch: current epoch at time of call
        """
        score = -val_loss

        # determine based on score
        if self.best_score is None:
            self.best_score = score
            save_here = True
        elif np.isnan(score):
            self.early_stop = True
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        # save checkpoint is stopping
        if self.early_stop:
            self.save_checkpoint(val_loss, model, epoch)

    def save_checkpoint(self, val_loss, model, epoch):
        """
        Saves model when validation loss decrease.

        :param float val_loss: avg loss from validation dataset
        :type nn.Module model: model from which to save early
        """
        if self.save_model == False:
            return
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        save_file = str(f"early_stop_model_ep_{epoch}_testloss_{val_loss:.4f}.pt")
        torch.save(model.state_dict(), os.path.join(self.path, save_file))

        self.val_loss_min = val_loss
