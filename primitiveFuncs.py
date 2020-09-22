import struct, math

def sq(x):
    return x**2


def safe_div(x, y):
    try:
        return x / y
    except ZeroDivisionError:
        return 0


def safe_sqrt(x):
    try:
        return math.sqrt(x)
    except ValueError:
        return 0