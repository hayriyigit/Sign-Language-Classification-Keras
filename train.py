import config
import numpy as np
import tensorflow as tf
from model import get_model
from dataset import DataLoader, Dataset
from sklearn.model_selection import train_test_split


def main():
    images = np.load(config.IMAGES_DATA)
    labels = np.load(config.LABELS_DATA)

    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=config.SPLIT_SIZE, shuffle=True)

    train_dataset = Dataset((x_train, y_train))
    train_loader = DataLoader(train_dataset, config.BATCH_SIZE, True)
    valid_dataset = Dataset((x_test, y_test))
    valid_loader = DataLoader(valid_dataset, config.BATCH_SIZE, True)

    model = get_model()
    if config.LOAD_MODEL:
        try:
            model.load_weights(config.CHECKPOINT_PATH)
        except tf.errors.NotFoundError as e:
            print("Checkpoint file not found!")
            print(e.message)

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(
                      learning_rate=config.LEARNING_RATE),
                  metrics=["accuracy"])

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=config.CHECKPOINT_PATH,
                                                     save_weights_only=True,
                                                     verbose=1)
    callbacks = []
    if config.SAVE_MODEL:
        callbacks.append(cp_callback)

    model.fit(train_loader,
              steps_per_epoch=len(train_loader),
              epochs=config.EPOCHS,
              verbose=1,
              validation_data=valid_loader,
              validation_steps=len(valid_loader),
              callbacks=callbacks)


if __name__ == "__main__":
    main()
