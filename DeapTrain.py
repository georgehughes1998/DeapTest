import operator
import random, math, time
import multiprocessing

import numpy as np
# import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms, gp

from realFunc import realFunc
from primitiveFuncs import *

with open("out.dat") as file:
    data = [l.split(";") for l in file.read().splitlines()]
    data = [(float(x1), float(x2), float(res1), float(res2)) for x1, x2, res1, res2 in data]

    points = [(x1, x2) for x1, x2, _, _ in data]
    realFuncDict = {(x1, x2): (res1, res2) for x1, x2, res1, res2 in data}


# Eval function
def evalSymbReg(individual, points, dim):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function
    try:
        sqerrors = ((func(x1, x2) - realFuncDict[x1, x2][dim]) ** 2 for (x1, x2) in points)
        return math.fsum(sqerrors) / len(points),
    except OverflowError:
        return 10000000000,


# Set up primitive set
pset = gp.PrimitiveSet("main", arity=2)
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

pset.renameArguments(ARG0="x1")
pset.renameArguments(ARG1="x2")

# Creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

def make_toolbox(dim=0):
    # Toolbox functions
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    # Toolbox
    toolbox.register("evaluate", evalSymbReg, points=points, dim=dim)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=40))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=40))

    return toolbox

toolbox = make_toolbox(dim=0)
toolbox2 = make_toolbox(dim=1)

# Statistics
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
mstats.register("avg", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

if __name__ == '__main__':
    population_size = 1000
    iterations = 8

    # Multiprocessing
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    # Variables
    hof_size = 2

    # Create and evolve population
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(hof_size)

    pop2 = toolbox2.population(n=population_size)
    hof2 = tools.HallOfFame(hof_size)


    # Reset log
    with open("log.log", "a") as logfile:
        logfile.write("-"*20 + "\n")

    # Train loop
    while True:
        start_time = time.time()

        # Train first func
        pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, iterations, stats=mstats,
                                           halloffame=hof, verbose=True)

        # Write pops
        sorted_pop = sorted(pop, key=lambda ind: ind.fitness, reverse=True)
        results = {str(gp.PrimitiveTree(i)): toolbox.evaluate(i) for i in sorted_pop}
        with open("solutions.csv", "w") as file:
            for r in results:
                file.write(f"{r}; {results[r]}\n")

        # Log first
        with open("log.log", "a") as logfile:
            logfile.write(f"{iterations} iterations completed for first func and saved in {time.time() - start_time}.\n")

        start_time = time.time()

        # Train second func
        pop2, log2 = algorithms.eaSimple(pop2, toolbox2, 0.5, 0.1, iterations, stats=mstats,
                                           halloffame=hof2, verbose=True)

        # Write second func
        sorted_pop = sorted(pop2, key=lambda ind: ind.fitness, reverse=True)
        results = {str(gp.PrimitiveTree(i)): toolbox2.evaluate(i) for i in sorted_pop}
        with open("solutions2.csv", "w") as file:
            for r in results:
                file.write(f"{r}; {results[r]}\n")

        # Log second
        with open("log.log", "a") as logfile:
            logfile.write(f"{iterations} iterations completed for second func and saved in {time.time() - start_time}.\n")


