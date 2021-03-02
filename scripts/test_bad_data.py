from struct import unpack
import os
from tensorflow.python.ops.gen_image_ops import decode_image
from tqdm import tqdm
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import tensorflow as tf

marker_mapping = {
    0xFFD8: "Start of Image",
    0xFFE0: "Application Default Header",
    0xFFDB: "Quantization Table",
    0xFFC0: "Start of Frame",
    0xFFC4: "Define Huffman Table",
    0xFFDA: "Start of Scan",
    0xFFD9: "End of Image",
}


class JPEG:
    def __init__(self, image_file):
        with open(image_file, "rb") as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while True:
            (marker,) = unpack(">H", data[0:2])
            # print(marker_mapping.get(marker))
            if marker == 0xFFD8:
                data = data[2:]
            elif marker == 0xFFD9:
                return
            elif marker == 0xFFDA:
                data = data[-2:]
            else:
                (lenchunk,) = unpack(">H", data[2:4])
                data = data[2 + lenchunk :]
            if len(data) == 0:
                break


bads = []

dir_ = sys.argv[1]


def custom_dataset(img_size: int, data_folder: str, channel_count: int):
    def load(file):

        img_data = tf.io.read_file(file)
        img = tf.io.decode_image(
            img_data,
            channels=channel_count,
            expand_animations=False,
            dtype=tf.float32,
        )

        if img.shape[0] != img_size:
            return tf.image.resize(img, [img_size, img_size])
        else:
            return img

    def normalize(x):
        return (x - 0.5) / 0.5

    ds = tf.data.Dataset.list_files(f"{data_folder}*")
    train_dataset = ds.map(
        load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    train_dataset = train_dataset.batch(256)
    train_dataset = train_dataset.map(
        normalize, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )
    # train_dataset = train_dataset.cache()
    return train_dataset.prefetch(tf.data.AUTOTUNE)


dataset = custom_dataset(32, data_folder=f"{dir_}*", channel_count=1)


for img in tqdm(os.scandir(dir_)):
    image = img.path

    tf_data = tf.io.read_file(image)

    if not tf.image.is_jpeg(tf_data):
        bads.append(img)
        continue

    try:
        tf.io.decode_image(tf_data, channels=1)
    except:
        bads.append(img)
        continue
        # print(f"bad image {img.name}")

    image = JPEG(image)

    try:
        image.decode()
    except:
        bads.append(img)


for name in bads:
    print(f"Removing {name.name}")
    os.remove(name.path)

for img in tqdm(dataset.take(256)):
    continue