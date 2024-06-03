import numpy as np
import math

C1 = math.sqrt(2)
C2 = 1.533751168755204288118041


def super_fibonacci_point(n, s, phi=C1, psi=C2):
    t = s/n
    d = 2*math.pi*s
    r = math.sqrt(t)
    R = math.sqrt(1-t)
    alpha = d/phi
    beta = d/psi
    point = [
        r*math.sin(alpha),
        r*math.cos(alpha),
        R*math.sin(beta),
        R*math.cos(beta)
    ]

    return point


def super_fibonacci(n, phi=C1, psi=C2):
    out = np.zeros((n,4))

    for i in range(n):
        s = i + 0.5
        point = super_fibonacci_point(n, s, phi=C1, psi=C2)
        out[i] = point

    return out