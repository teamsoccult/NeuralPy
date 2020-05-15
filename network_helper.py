'''
TASKS: P, Q 
'''

import random
import math_helper as M
import read_write_helper as RW

### TASK P)

def create_batches(values, batch_size):
    '''Using the random.shuffle function from the random module,
    this function partitions a list of values into random batches of
    length batch_size. The only exception is the last batch, which can be
    of a smaller length.
    Assumes that the input is a list and that batch_size is an integer.
    Returns a list of batches of values.
    '''
    values_list = []
    values_copy = values[:]
    random.shuffle(values_copy)
    current_batch = 0

    while current_batch < len(values_copy):
        current_batch += batch_size
        values_list.append(values_copy[current_batch-batch_size:current_batch])

    return values_list

### TASK Q)

def update(network, images, labels, sigma = 0.1):
    A, b = network
    A_list = [[0]*len(network[1]) for i in range(len(A))]
    b_list = [[0 for i in range(len(b))]]

    for n in range(len(images)):
        x = RW.image_to_vector(images[n])
        a = M.predict(network, x)
        y = M.categorical(labels[n])

        for j in range(len(b)):

            current_element = 2 * (a[j] - y[j]) / 10

            b_list[0][j] += current_element

            for i in range(len(A)):
                A_list[i][j] += x[i] * current_element

    b_list_final = M.scalar_multiplication(b_list, (sigma * 1/len(images)))
    b = M.sub([b], b_list_final)

    A_list_final = M.scalar_multiplication(A_list, (sigma * 1/len(images)))
    A = M.sub(A, A_list_final)

    network = [A, b[0]]

    return network

### TASK R)

def learn(images, labels, epochs, batch_size):
    #initializing the random network:
    b = [random.uniform(0, 1) for m in range(10)]
    A = [[random.uniform(0, 1/784) for n in range(10)] for n in range(784)]
    print(f"b dim {M.dim(b)}")
    print(f"A dim {M.dim(A)}")
    network = [A, b]
    print(f"network dim {M.dim(network)}")

    for e in range(epochs):
        batches = create_batches(list(range(len(images))), batch_size)
        for i in batches: #this should be smarter..
            one_img_batch = [images[j] for j in i]
            one_lab_batch = [labels[j] for j in i]
            print(f"one img batch {M.dim(one_img_batch)}")
            print(f"one lab batch {M.dim(one_lab_batch)}")
            network = update(network, one_img_batch, one_lab_batch, sigma = 0.1)
            pred, cost, acc = M.evaluate(network, images, labels)
            print(f"cost {cost}")
            print(f"acc {acc}")