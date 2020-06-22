# TensorFlow and tf.keras
import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import numpy
import h5py

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

def covert_tflite(model_name):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_name)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_model = converter.convert()
    model_name_lite = model_name
    model_name_lite += '_lite.tflite'
    open(model_name_lite, "wb").write(quantized_model)

def export_mode(model_name):
    LOG_DIR = 'logs'
    with tf.Session() as sess:
        model_filename = 'nnmodel.pb'  # your model path
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
    train_writer = tf.summary.FileWriter(LOG_DIR)
    train_writer.add_graph(sess.graph)

def predict_cloth(model_name, image, correct_label):
    model = tf.keras.models.load_model(model_name)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    show_image(image, class_names[correct_label], predicted_class)

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Excpected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def get_number():
    while True:
        num = input("Pick a number: ")
        if num.isdigit():
            num = int(num)
            if 0 <= num <= 1000:
                return int(num)
        else:
            print("Try again...")

def load_data():
    fashion_mnist = keras.datasets.fashion_mnist  # load dataset
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into tetsing and training
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, test_images, train_labels, test_labels

train_images, test_images, train_labels, test_labels = load_data()
#num = get_number()
#image = test_images[num]
#label = test_labels[num]
#predict_cloth('nnmodel', image, label)
#covert_tflite('nnmodel')


import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

import tensorflow as tf
from tensorflow.python.platform import gfile
with tf.Session() as sess:
    model_filename ='saved_model.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
LOGDIR='/logs/tests/1/'
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)