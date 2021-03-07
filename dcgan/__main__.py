"""Console script for tf_examples."""
import sys
import fire
import os


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

from dcgan import *


def main(epochs, logname, channels=1, batch_size=256, data_folder=None):
    # """Console script for dcgan."""

    if data_folder:
        os.environ["DCGAN_DATADIR"] = data_folder

    os.environ["DCGAN_CHANNEL"] = str(channels)

    import dcgan.dcgan

    obj = dcgan.dcgan.DCGAN(epochs, logname, channels, batch_size, data_folder)
    obj.train()

    return 0


def fire_():
    fire.Fire(main)