'''
TASKS: R
'''
import read_write_helper as RW
import network_helper as NH
import math_helper as M
import plots_helper as P
import random
import matplotlib.pyplot as plt

### IMPORT FILES 

## DICTIONARIES FOR TEST AND TRAINING
filename_test = {'images' : 't10k-images.idx3-ubyte' ,'labels' : 't10k-labels.idx1-ubyte'}
filename_train = {'images' : 'train-images.idx3-ubyte' ,'labels' : 'train-labels.idx1-ubyte'}

## READ TRAINING DATA

labels = RW.read_labels(filename_train['labels'])
images = RW.read_image(filename_train['images'])

## READ NETWORK

network = RW.linear_load('mnist_linear.weights')

## CONSTANTS FOR LEARNING

batch_size = 100
epochs = 5

## LEARN AND SAVE "fast_network"

RW.linear_save("fast_network", NH.fast_learn(images, labels, epochs, batch_size))

## LOAD "fast_network"

fast_network = RW.linear_load("fast_network")

## READ TEST DATA

labels = RW.read_labels(filename_test['labels'])
images = RW.read_image(filename_test['images'])

## EVALUATE THE NETWORK

M.evaluate(fast_network, images, labels)

## PLOT THE NETWORKS

#Initialize untrained network
untrained_network = [[random.uniform(0, 1/784) for n in range(10)] for n in range(784)]

# SAVE THE PLOTS

P.weights_plot(network[0]).savefig("images/linear_network.png")
P.weights_plot(fast_network[0]).savefig("images/fast_network.png")
P.weights_plot(untrained_network).savefig("images/untrained_network.png")