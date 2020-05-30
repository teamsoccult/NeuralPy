
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import read_write_helper as RW
import network_helper as NH
import math_helper as M
import plots_helper as P
import random

import matplotlib
import matplotlib.pyplot as plt

import matplotlib.animation as animation

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()

untrained_network = [[random.uniform(0, 1/784) for n in range(10)] for n in range(784)]

im = P.weights_plot(untrained_network)

def init():
    im = P.weights_plot(untrained_network)
    return [im]

# animation function.  This is called sequentially
def animate(i):
    a=im.get_array()
    a=a*np.exp(-0.001*i)    # exponential decay of the values
    im.set_array(a)
    return [im]

anim = animation.FuncAnimation(fig, animate_func)

#anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

plt.show()  # Not required, it seems!