# %%
import numpy as np
import torch
from utils.diffuser_utils import load_mask
import train as train
import matplotlib.pyplot as plt

import PIL.Image as Image
import dataset as ds
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ[
    "CUDA_VISIBLE_DEVICES"
] = "0,1,2,3"  # try notebook restart, try 'export CUDA_VISIBLE_DEVICES=0,1,2,3"
# %%
torch.cuda.device_count()
torch.cuda.get_device_name(0)
# %%
train_dataloader, test_dataloader, val_dataloader = train.get_data()
model = train.get_model(stack_depth=train.STACK_DEPTH, device="cuda:0")

model_dir = "saved_models/checkpoint_4_3_symmetric_False/saved_model_ep230_testloss_0.001096144231269136.pt"
model.load_state_dict(torch.load(model_dir))
# %%
plt.imshow(np.mean(load_mask(), -1))
# %%

predictions = []
inputs = []
for sample in val_dataloader:
    inputs.append(sample["image"])
    prediction = model(sample["image"].to(train.DEVICE))
    predictions.append(prediction)
    torch.cuda.empty_cache()
    break

for i in range(len(predictions)):
    pred = predictions[i].detach().cpu().numpy()
    pred_im = np.mean(pred[0], 0)
    input_ = inputs[i].detach().cpu().numpy()
    input_im = np.mean(input_[0], 0)

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(input_im)
    ax[0].set_title("prediction")
    ax[1].imshow(pred_im)
    ax[0].set_title("ground")
    plt.show()
# %%
plt.imshow(np.mean(model.output1[0, 2].detach().cpu().numpy(), 0), cmap="gray")


# %%
class ImagePredictor:
    def __init__(self, model, measurements):
        self.model = model
        self.measurements = measurements

    def predict_measurement(self, measurements=None):
        if measurements == None:
            measurements = self.measurements

        meas_array = np.transpose(
            np.array(measurements)[:, 200:456, 200:456], (1, 2, 0)
        )
        meas_array = {"image": meas_array}

        self.meas_tensor = ds.toTensor()(meas_array)
        self.meas_tensor["image"] = self.meas_tensor["image"].unsqueeze(0)
        print("Predicting measurements of size: ", self.meas_tensor["image"].shape)

        self.meas_pred = self.model.forward(
            self.meas_tensor["image"].to(train.DEVICE), spec_dim=2, sim=False
        )
        self.meas_pred = self.meas_pred.detach().cpu().numpy()
        plt.imshow(np.mean(self.meas_pred[0], 0))
        plt.show()


# %%
focused_25 = np.asarray(
    Image.open(
        "../defocuscamdata/test_measurements/test_measurements_2/duck_focused_300.png.png"
    )
)
blurry_1_100 = np.asarray(
    Image.open(
        "../defocuscamdata/test_measurements/test_measurements_2/duck_blurry_1_100.png.png"
    )
)
blurry_2_300 = np.asarray(
    Image.open(
        "../defocuscamdata/test_measurements/test_measurements_2/duck_blurry_2_300.png.png"
    )
)
measurements = [focused_25, blurry_2_300, blurry_1_100]

predictor = ImagePredictor(model, measurements)
predictor.predict_measurement(measurements)
# %%
predictor.model.output1.shape
# %%
plt.imshow(predictor.meas_tensor["image"][0, 2].detach().cpu().numpy())
plt.show()
# plt.imshow(np.mean(predictor.model.output1[0,2].detach().cpu().numpy(),0))
# plt.show()
# %%
