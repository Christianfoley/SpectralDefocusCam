# Credit to : https://github.com/basler/pypylon/tree/master/samples
# for pypylon samples
import os
import numpy as np
import time
from opto import Opto
from pypylon import pylon
from PIL import Image

LENS_PORT = ""  # TODO  # Input a port here
MODE = "1"  # 0 for capture 1, 1 for capture 1 series (blur increments)
TIME_DELAY = 1
CURRENT_LEVELS = [81.24, 76.45, 71.08]
EXPOSURE_TIME = 1
SAVE_DIR = ""  # TODO  # Input a save location here
CYCLES = 100  # Number of cycles for continuous capture (abt 0.8s per cycle)


def main():
    # connectto the lens
    lens = Opto(port=LENS_PORT)
    lens.connect()

    # connect to the camera, start from "power-on" state
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    camera.UserSetSelector = "Default"
    camera.UserSetLoad.Execute()
    camera.ExposureTime = EXPOSURE_TIME

    print("Starting capture:")
    if MODE == 0:
        capture = capture_one(camera, lens, current_level=CURRENT_LEVELS[0])
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
    elif MODE == 1:
        series = capture_series(camera, lens, CURRENT_LEVELS)
        save_series(series, SAVE_DIR, CURRENT_LEVELS)
    elif MODE == 2:
        series = capture_series_cont(camera, lens, SAVE_DIR, CURRENT_LEVELS, CYCLES)
    else:
        raise AttributeError(f"Mode {MODE} not accepted. Must be one of (0,1,2)")
    print("Done!")

    lens.close()
    camera.Close()


def grab_samples(camera, num_samples):
    """
    Grabs num_samples images an returns their arrays.
    Images are of dimensionality:
        (camera.Height.Value, camera.Width.Value)

    :param pylon.InstantCamera camera: open camera object
    :param int num_samples: number of samples to retrieve

    :return list[np.ndarray] captures: Returns captures as numpy arrays, uint16
    """
    # fetch some images with foreground loop
    captures = []
    for i in range(num_samples):
        with camera.GrabOne(1000) as res:
            img = res.Array
            captures.append(img)

    return captures


# --------------------------- single capture ---------------------------#
def capture_one(camera, lens, current_level):
    # demonstrate some feature access
    new_width = camera.Width.GetValue() - camera.Width.GetInc()
    if new_width >= camera.Width.GetMin():
        camera.Width.SetValue(new_width)

    lens.current(current_level)
    time.sleep(0.2)  # wait for lens to focus

    capture = grab_samples(camera, 1)[0]

    return capture


# --------------------------- series capture ---------------------------#
def capture_series(camera, lens, current_levels):
    """
    Capture a blur series, with one capture at each blur level
    """
    series = []
    for current_level in current_levels:
        capture = capture_one(camera, lens, current_level)
        series.append(capture)

    return series


# --------------------- continuous series capture ----------------------#
def capture_series_cont(camera, lens, save_location, current_levels, cycles=100):
    """
    Continuously capture blur series, with one capture at each blur level,
    one cycle of captures at a time.

    Continuously saves results to save_location
    """
    if not os.path.exists(save_location):
        os.mkdir(save_location)

    for i in range(cycles):
        print(f"Capturing series: {i}/{cycles}", end="\r")
        series = capture_series(camera, lens)

        save_series(series, save_location, current_levels, name=str(i))


def save_series(series, save_location, current_levels, name=""):
    """
    save a series at a given save location
    """
    if name:
        name = "_" + name

    save_dir = os.path.join(save_location, f"series{name}")
    if not os.exists(save_dir):
        os.mkdir(save_dir)

    for i, capture in enumerate(series):
        image = Image.fromarray(capture.astype(np.uint8))
        image.save(os.path.join(save_dir, f"current_{current_levels[i]}.png"), "PNG")


if __name__ == "__main__":
    main()
