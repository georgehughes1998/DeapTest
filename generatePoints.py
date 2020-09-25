import numpy as np
import random
from realFunc import realFunc

NUM_LINES = 10000
XRANGE = -3, 3
YRANGE = -3, 3

with open("out.dat", "w") as file:
    for l in range(NUM_LINES):
        # x, y = random.uniform(*XRANGE), random.uniform(*YRANGE)
        x1, x2 = random.uniform(*XRANGE), random.uniform(*XRANGE)
        res1, res2 = realFunc(x1, x2)
        file.write(f"{x1}; {x2}; {res1}; {res2}\n")


# with open("out.dat") as file:
#     points = [l.split(",") for l in file.read().splitlines()]
#     points = [(float(x), float(y), int(res)) for x, y, res in points]
# print(points)