import tensorflow as tf

import matplotlib.pyplot as plt
import os
import time
import datetime

from dcgan.discriminator import make_discriminator_model, discriminator_loss
from dcgan.generator import make_generator_model, generator_loss
from dcgan.dataset import make_dataset

from dcgan.utils import *
from dcgan.metrics import *
from dcgan import CHECKPOINT_DIR, MODEL_DIR


channel_count = int(os.environ["DCGAN_CHANNEL"])

tf.profiler.experimental.server.start(6009)

generator = make_generator_model(32, channel_count)

discriminator = make_discriminator_model(32, channel_count)

generator_optimizer = tf.keras.optimizers.Adam(1e-04, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-04, beta_1=0.5)
dataset = make_dataset(32, os.environ["DCGAN_DATADIR"], channel_count)


class DCGAN(object):
    def __init__(self, **kwargs) -> None:

        self.batch_size = kwargs["batch_size"]
        self.noise_dim = 100
        self.num_examples_to_generate = 16
        self.img_size = 28
        self.data_folder = kwargs["data_folder"]
        self.num_channels = kwargs["channels"]
        self.epochs = kwargs["epochs"]
        self.lr = 1e-4
        self.lr2 = 1e-4
        self.batch_size = 256
        self.train_summary_writer = make_summary_writer(**kwargs)

        show_dataset(dataset, self.num_examples_to_generate, self.train_summary_writer)

        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(1),
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            generator=generator,
            discriminator=discriminator,
        )

        self.manager = tf.train.CheckpointManager(
            self.checkpoint, CHECKPOINT_DIR, max_to_keep=3
        )

    def train(self):

        self.checkpoint.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:

            print("Restored from {}".format(self.manager.latest_checkpoint))
            current_step = int(self.checkpoint.step.numpy())
            print(
                f"Continuing from epoch {current_step} + {self.epochs} -> {self.epochs + current_step}"
            )
            epochs = range(current_step, self.epochs + current_step)
        else:
            epochs = range(self.epochs)
            print("Initializing from scratch.")

        for epoch in epochs:
            seed = tf.random.normal([self.num_examples_to_generate, self.noise_dim])
            start = time.time()
            fake_out.reset_states()
            real_out.reset_states()
            gen_loss_metric.reset_states()
            disc_loss_metric.reset_states()

            # data = iter(dataset.take(256).cache())
            # print(dir(data))
            # print(data.output_shapes)
            # for step in range(256):

            #     with tf.profiler.experimental.Trace("train", step_num=step, _r=1):
            #         self.train_step(next(data), epoch)

            #     if step == 255:
            #         print("\n")
            #     elif step % 2 == 0:
            #         print(".", end="", flush=True)

            for step, batch in enumerate(dataset.take(256)):
                self.train_step(batch, epoch)

            self.checkpoint.step.assign_add(1)
            if int(self.checkpoint.step) % 15 == 0:
                save_path = self.manager.save()
                print(
                    "Saved checkpoint for step {}: {}".format(
                        int(self.checkpoint.step), save_path
                    )
                )

                # Produce images for the GIF as we go
            generate_and_save_images(
                generator, epoch + 1, seed, self.train_summary_writer
            )
            print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

        # Generate after the final epoch
        generate_and_save_images(
            generator, self.epochs, seed, self.train_summary_writer
        )

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        generator.save(os.path.join(MODEL_DIR, f"gen_trained_{current_time}"))
        discriminator.save(os.path.join(MODEL_DIR, f"disc_trained_{current_time}"))

    @tf.function
    def train_step(self, images, epoch):
        noise = tf.random.normal([self.batch_size, self.noise_dim])
        # tf.random.gau
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            # gen_loss = tf.vectorized_map(generator_loss, fake_output)
            # disc_loss = tf.vectorized_map(discriminator_loss, fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            gen_loss_metric.update_state(gen_loss)
            disc_loss_metric.update_state(disc_loss)
            fake_out.update_state(fake_output[0])
            real_out.update_state(real_output[0])

        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )

        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )

        record_metrics(epoch, self.train_summary_writer)