'''
TASKS: R
'''
import read_write_helper as RW
import network_helper as NH
import math_helper as M

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