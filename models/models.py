from models.Unet.babyunet import BabyUnet
from models.conv.dncnn import DnCNN
from models.conv.singleconv import SingleConvolution
from models.Unet.unet import Unet
from models.LCNF.liif import LIIF


def get_model(name, in_channels, out_channels, **kwargs):
    if name == "unet":
        return Unet(in_channels, out_channels)
    if name == "baby-unet":
        return BabyUnet(in_channels, out_channels)
    if name == "dncnn":
        return DnCNN(in_channels, out_channels)
    if name == "convolution":
        return SingleConvolution(in_channels, out_channels, kwargs["width"])
    if name == "lcnf":
        return LIIF()
