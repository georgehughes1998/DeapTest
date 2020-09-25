import struct, math

def sq(x):
    try:
        return x**2
    except OverflowError:
        return 1000000000000


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
