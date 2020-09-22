Z_POWER = 2
DEPTH = 10000
SCALE = 1
XOFFSET, YOFFSET = 0, 0

# The function to approximate
def realFunc(x, y):
    z = 0
    c = x + y*1j

    for i in range(DEPTH, 0, -1):
        if z.real > 2:
            return i
        else:
            z = (z ** Z_POWER) + (c/SCALE) + XOFFSET + (YOFFSET*1j)
    return 0