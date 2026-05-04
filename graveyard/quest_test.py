import tensorflow as tf
import logging
import os
import datetime

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    logging.info("Hello World")
    logging.info("Current working directory: {}".format(os.getcwd()))
    logging.info("Current time: {}".format(datetime.datetime.now()))
    logging.info("TensorFlow version: {}".format(tf.__version__))
    logging.info("Num GPUs Available: {}".format(len(tf.config.list_physical_devices('GPU'))))
    logging.info("Num GPUs Available: {}".format(tf.config.experimental.list_physical_devices('GPU')))
    logging.info("Num GPUs Available: {}".format(tf.config.list_logical_devices('GPU')))