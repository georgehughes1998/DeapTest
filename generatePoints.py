import numpy as np
import random
from realFunc import realFunc

NUM_LINES = 10000
XRANGE = -2, 0.5
YRANGE = -0.5, 0.5

with open("out.dat", "w") as file:
    for l in range(NUM_LINES):
        x, y = random.uniform(*XRANGE), random.uniform(*YRANGE)
        res = realFunc(x, y)
        file.write(f"{x}, {y}, {res}\n")


# with open("out.dat") as file:
#     points = [l.split(",") for l in file.read().splitlines()]
#     points = [(float(x), float(y), int(res)) for x, y, res in points]
# print(points)