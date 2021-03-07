"""Console script for tf_examples."""
import sys
import click

import os


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

from dcgan import *

# from tf_examples.lin_reg import lin_reg


@click.command()
@click.option("--epochs", type=click.INT, default=50, help="Default 50")
@click.option("--logname", type=click.STRING, required=True)
@click.option("--channels", type=click.INT, default=1, help="Number of color channels")
@click.option("--batch-size", type=click.INT, default=256)
@click.option(
    "--data-folder", type=click.STRING, help="Raw images any size", required=False
)
def main(**kwargs):
    # """Console script for dcgan."""

    if kwargs["data_folder"]:
        os.environ["DCGAN_DATADIR"] = kwargs["data_folder"]

    os.environ["DCGAN_CHANNEL"] = str(kwargs["channels"])

    import dcgan.dcgan

    obj = dcgan.dcgan.DCGAN(**kwargs)
    obj.train()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
