# %%
import utils.helper_functions as helper
import torch
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# %%
device = "cuda:0"
a = torch.rand(5, 5, 5)
a = a.cuda()
# %%
torch.cuda.get_device_name(a.device)
# %%
