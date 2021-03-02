import tensorflow as tf

import matplotlib.pyplot as plt
import os
import time
import datetime

from dcgan.discriminator import make_discriminator_model, discriminator_loss
from dcgan.generator import make_generator_model, generator_loss
from dcgan.dataset import make_dataset

from dcgan.utils import make_gif, plot_to_image, generate_and_save_images
from dcgan import CHECKPOINT_DIR, MODEL_DIR


def main(**kwargs):
    gen_learning_rate = kwargs["lr"]

    disc_learning_rate = kwargs["lr2"]
    img_size = kwargs["img_size"]

    EPOCHS = kwargs["epochs"]

    data_folder = kwargs["data_folder"]
    noise_dim = 100
    num_examples_to_generate = 16
    BATCH_SIZE = 256

    train_dataset = make_dataset(img_size, data_folder, kwargs["channels"])
    generator = make_generator_model(img_size, kwargs["channels"])
    discriminator = make_discriminator_model(img_size, kwargs["channels"])

    print(f"Generator input shape: {generator.input_shape}")
    print(f"Generator output shape: {generator.output_shape}")
    print(f"Discriminator input shape: {discriminator.input_shape}")
    print(f"Discriminator output shape: {discriminator.output_shape}")
    fig = plt.figure(figsize=(4, 4))

    for i, img in enumerate(train_dataset.take(num_examples_to_generate)):

        if i == 0:
            print(
                f"1 data sample has value range from {tf.reduce_min(img).numpy()} to {tf.reduce_max(img).numpy()}"
            )

        plt.subplot(4, 4, i + 1)
        plt.imshow(img[i, :, :, :] * 0.5 + 0.5, cmap="gray")
        plt.axis("off")

    if kwargs["show"]:
        plt.show()
