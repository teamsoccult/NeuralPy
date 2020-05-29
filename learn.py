'''
TASKS: R
'''
import read_write_helper as RW
import network_helper as NH
import math_helper as M


filename_test = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 't10k-labels.idx1-ubyte'}
filename_train = {'images' : 'train-images.idx3-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}
labels = RW.read_labels(filename_train['labels'])
images = RW.read_image(filename_train['images'])
network = RW.linear_load('mnist_linear.weights')
batch_size = 100
epochs = 5

RW.linear_save("fast_network", NH.fast_learn(images, labels, epochs, batch_size))

NH.learn(images, labels, epochs, batch_size)

RW.linear_save("trained_network", NH.learn_CE(images, labels, epochs, batch_size))

M.evaluate(RW.linear_load("trained_network"), filename_test['images'], filename_test['labels'])