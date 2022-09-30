import numpy as np
import matplotlib.pyplot as plt
import pickle


with open("Data set/pickled_mnist.pkl", "br") as fh:
    data = pickle.load(fh)

image_size = 28
labels = 10
image_pixels = image_size * image_size

lr = np.arange(labels)

train_images = data[0]
test_images = data[1]
train_labels = data[2]
test_labels = data[3]

train_labels_one_hot = (lr == train_labels).astype(np.float64)
test_labels_one_hot = (lr == test_labels).astype(np.float64)
