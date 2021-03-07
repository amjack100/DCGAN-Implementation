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
@click.option("--lr", type=click.FLOAT, default=1e-4, help="Generator learning rate")
@click.option(
    "--lr2", type=click.FLOAT, default=1e-4, help="Discriminator learning rate"
)
@click.option("--img-size", type=click.INT, default=28, help="Both width and height")
@click.option("--logname", type=click.STRING, required=True)
@click.option("--reset", flag_value=True)
@click.option("--show", flag_value=True, help="Use imshow")
@click.option("--channels", type=click.INT, default=1, help="Number of color channels")
@click.option("--batch-size", type=click.INT, default=256)
@click.option(
    "--whatif",
    flag_value=True,
    help="Display all training info without actually training",
)
@click.option("--data-glob", type=click.STRING, help="Raw images any size")
def main(**kwargs):
    # """Console script for dcgan."""

    if kwargs["data_folder"]:
        os.environ["DCGAN_DATADIR"] = kwargs["data_folder"]

    os.environ["DCGAN_CHANNEL"] = str(kwargs["channels"])

    if kwargs["reset"]:
        for item in os.scandir(CHECKPOINT_DIR):
            os.remove(item.path)
        for item in os.scandir(IMAGE_DIR):
            os.remove(item.path)
        return

    import dcgan.dcgan

    # import dcgan.whatif

    # if kwargs["whatif"]:
    #     dcgan.whatif.main(**kwargs)
    # else:
    obj = dcgan.dcgan.DCGAN(**kwargs)
    obj.train()

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
