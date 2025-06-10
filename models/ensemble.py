import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.diffuser_utils import *


class SSLSimulationModel(nn.Module):
    """
    Combines a forward measurement simulation model and a reconstruction model
    into a single end-to-end simulation for a bit of self-supervised syntactic
    sugar.
    """
    def __init__(self, model1, model2):
        super(SSLSimulationModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        self.output1 = self.model1(x)
        self.output2 = self.model2(self.output1)

        if self.model1.operations["spectral_pad"]:
            self.output2 = self.output2[:, 1:-1, :, :]
        return self.output2
