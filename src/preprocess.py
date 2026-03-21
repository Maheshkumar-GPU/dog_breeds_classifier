import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from config import dataset_path, IMAGE_SIZE, BATCH_SIZE

def load_dataset():
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_dataset.class_names


    train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
    val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y))



    return train_dataset, val_dataset, class_names