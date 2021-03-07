"""Console script for tf_examples."""
import sys
import fire
import os


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

from dcgan import *


def main(epochs, logname, channels=1, batch_size=256, data_folder=None):
    # """Console script for dcgan."""

    CHECKPOINT_DIR = f"{os.path.split(__file__ )[0]}/../checkpoints"
    IMAGE_DIR = f"{os.path.split(__file__ )[0]}/../images"
    MODEL_DIR = f"{os.path.split(__file__ )[0]}/../models"

    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)

    if not os.path.exists(IMAGE_DIR):
        os.mkdir(IMAGE_DIR)

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if data_folder:
        os.environ["DCGAN_DATADIR"] = data_folder

    os.environ["DCGAN_CHANNEL"] = str(channels)

    import dcgan.dcgan

    obj = dcgan.dcgan.DCGAN(epochs, logname, channels, batch_size, data_folder)
    obj.train()

    return 0


def fire_():
    fire.Fire(main)