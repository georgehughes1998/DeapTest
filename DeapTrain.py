import operator
import random, math
import multiprocessing

import numpy as np
# import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms, gp

from realFunc import realFunc
from primitiveFuncs import *

with open("out.dat") as file:
    data = [l.split(",") for l in file.read().splitlines()]
    data = [(float(x), float(y), int(res)) for x, y, res in data]

    points = [(x, y) for x, y, _ in data]
    realFuncDict = {(x, y): res for x, y, res in data}


# Eval function
def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function
    try:
        sqerrors = ((func(x, y) - realFuncDict[x, y]) ** 2 for (x, y) in points)
        return math.fsum(sqerrors) / len(points),
    except OverflowError:
        return 10000000,






# Set up primitive set
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
# pset.addTerminal(1)

pset.addEphemeralConstant("Uniform", lambda: random.uniform(-1, 1))
pset.addEphemeralConstant("Randint", lambda: random.randint(-10, 10))

pset.renameArguments(ARG0="x")
pset.renameArguments(ARG1="y")

# Creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

# Toolbox functions
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Toolbox
toolbox.register("evaluate", evalSymbReg, points=points)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=40))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=40))

# Statistics
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)


if __name__ == '__main__':
    iterations = 100

    # Multiprocessing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # Variables
    hof_size = 2

    # Create and evolve population
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(hof_size)

    # for e in range(1000):
    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, iterations, stats=mstats,
                                       halloffame=hof, verbose=True)

    sorted_pop = sorted(pop, key=lambda ind: ind.fitness, reverse=True)
    results = {str(gp.PrimitiveTree(i)): toolbox.evaluate(i) for i in sorted_pop}
    with open("solutions.csv", "w") as file:
        for r in results:
            file.write(f"{r}; {results[r]}\n")

        # print()
        #
        # # Draw image
        # fig, axs = plt.subplots(3, 2)
        #
        # for mi in range(hof_size):
        #     tree = gp.PrimitiveTree(hof[mi])
        #     function = gp.compile(tree, pset)
        #
        #     m = np.zeros(shape=(256, 256))
        #     for xi in range(256):
        #         for yi in range(256):
        #             x, y = (yi / 128) - 1.5, (xi / 128) - 1
        #             m[xi, yi] = function(x, y)
        #     # Display matrix
        #     i, j = mi % 3, mi // 3
        #     axs[i, j].imshow(m)
        #     axs[i, j].set_title(f"({i}, {j})")
        #     print(f"({i}, {j}): {tree}")
        #     axs[i, j].set_xticks([])
        #     axs[i, j].set_yticks([])

        # plt.show()
