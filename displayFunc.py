import random
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms, gp
import operator
import numpy as np
from primitiveFuncs import *


SHOW_WORST = False


with open("solutions.csv") as file:
    solutions = [s.split(";") for s in file.read().splitlines()]
if SHOW_WORST:
    solutions.reverse()


pset = gp.PrimitiveSet("main", 2)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safe_div, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(safe_sqrt, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.tan, 1)
pset.addPrimitive(sq, 1)

# pset.addEphemeralConstant("Uniform", lambda: random.uniform(-1, 1))
# pset.addEphemeralConstant("Randint", lambda: random.randint(-10, 10))

pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")


xsize, ysize = 3, 3
solutions_grid = [[(solutions[x + y*ysize][0], x + y*ysize) for x in range(ysize)] for y in range(xsize)]
fig, axs = plt.subplots(xsize, ysize)

scale = 64
zoom = 2

for plotx in range(xsize):
    for ploty in range(ysize):
        f, score = solutions_grid[plotx][ploty]

        tree = gp.PrimitiveTree.from_string(f, pset)
        function = gp.compile(tree, pset)

        m = np.zeros(shape=(scale, scale))
        for xi in range(scale):
            for yi in range(scale):
                x, y = (yi / (scale/zoom)) - 1.5, (xi / (scale/zoom)) - 1
                m[xi, yi] = function(x, y)
        axs[plotx, ploty].imshow(m)
        axs[plotx, ploty].set_title(str(score))
        axs[plotx, ploty].set_xticks([])
        axs[plotx, ploty].set_yticks([])

bestworst = ["Best", "Worst"][SHOW_WORST]
fig.suptitle(f"{bestworst} Tree GP Solutions")
plt.savefig("plot.png")
plt.show()