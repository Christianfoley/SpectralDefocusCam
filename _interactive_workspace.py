# %%
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

from models_learning.LCNF.liif import LIIF
import models_learning.LCNF.edsr
import models_learning.LCNF.mlp

from utils.lcnf_utils import make_coord, batched_predict
from utils.helper_functions import read_config, plt3D

import data_utils.dataset as ds
import inference

encoder_specs = [
    {
        "name": "edsr",
        "args": {
            "no_upsampling": True,
            "input_channel": 30,
        },
    }
] * 5
imnet_spec = {
    "name": "mlp",
    "args": {
        "out_dim": 30,
        "hidden_list": [256, 256, 256, 256, 256],
    },
}

device_ID = 1
device = f"cuda:{device_ID}"
torch.cuda.get_device_name(device_ID)

# %%
# ---- LOAD MODEL ---- #
weights = "/home/cfoley_waller/defocam/defocuscamdata/models/checkpoint_4_5_True_symmetric_False/2023_05_09_20_44_31/saved_model_ep100_testloss_0.0013107480894847725.pt"
train_config = read_config(
    "/home/cfoley_waller/defocam/SpectralDefocusCam/config_files/training/train_5_9_2023_stack5_sim_blur.yml"
)
spectral_ensemble = inference.get_model_pretrained(
    weights=weights, train_config=train_config, device=device
).to(device)

model = LIIF(encoder_specs, imnet_spec=imnet_spec).to(device)
model
# %%
# ---- LOAD SIMULATED DATA ---- #
_, _, test_loader = ds.get_data(
    batch_size=1,
    data_split=train_config["data_partition"],
    base_path=train_config["base_data_path"],
    workers=1,
)
for sample in test_loader:
    x = sample["image"].to(device)
    x = spectral_ensemble.model1(x, sim=True)
    print(x.shape)
    break

# %%
# ---- RUN TEST INFERENCE ---- #

temp = x[:, :, :, :64, :64]
h, w = 64, 64
coord = make_coord((h, w)).to(device)
cell = torch.ones_like(coord)
cell[:, 0] *= 2 / h
cell[:, 1] *= 2 / w

pred = batched_predict(
    model,
    temp,
    coord.unsqueeze(0),
    cell.unsqueeze(0),
    bsize=30000,
)[0]
pred = (pred * 1 + 0).clamp(0, 1).view(h, w, 30).permute(2, 0, 1).cpu()
print(pred.shape)
plt3D(pred.detach().cpu().numpy().transpose(1, 2, 0))

# %%
pred = model(x[:, :, :, :64, :64])
pred = (pred * 1 + 0).clamp(0, 1).view(64, 64, 30).permute(2, 0, 1).cpu()
print(pred.shape)
plt3D(pred.detach().cpu().numpy().transpose(1, 2, 0))
# %%
# ---- LOAD TEST IMAGE ---- #
# test_input_path = "/home/cfoley_waller/defocam/LCNF/Dataset/Inference/22.npy"

# img = np.load(test_input_path).astype("float32")
# if len(img.shape) == 2:
#     img = np.expand_dims(img, axis=0)
# img = torch.tensor(img)
# temp = img
# print(temp.shape)
