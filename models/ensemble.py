import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.diffuser_utils import *


class MyEnsemble(nn.Module):
    def __init__(self, model1, model2):
        super(MyEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        self.output1 = self.model1(x)
        self.output2 = self.model2(self.output1)

        if self.model1.operations["spectral_pad"]:
            self.output2 = self.output2[:, 1:-1, :, :]
        return self.output2
