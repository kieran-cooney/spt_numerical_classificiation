"""Functions for generating super fibonacci spirals. Implements functions
described in this paper:
https://openaccess.thecvf.com/content/CVPR2022/papers/Alexa_Super-Fibonacci_Spirals_Fast_Low-Discrepancy_Sampling_of_SO3_CVPR_2022_paper.pdf

The objective is to generate a collection of points over the 3 dimensional
sphere (or SO(3)) that are as seperated from each other as possible.
"""
import numpy as np
import math

# Constants as defined in the paper.
C1 = math.sqrt(2)
C2 = 1.533751168755204288118041


def super_fibonacci_point(n, s, phi=C1, psi=C2):
    """
    Generate a single point of a super fibonacci spiral.
    See paper for formulae and justification.

    Paramters
    ---------
    n : int
        The number of  points  in the spiral
    s : float
        A real number parametrising the points on the spiral
    phi: float
        A constant affecting the shape of the spiral in the first two
        coordinates
    psi: float
        A constant affecting the shape of the spiral in the second two
        coordinates

    Returns
    -------
    list of 4 floats
        The coordinates of the 4 dimensional super fibonacci spiral point on
        the 3 dimensional sphere
    """
    t = s/n
    d = 2*math.pi*s

    # Radial variables
    r = math.sqrt(t)
    R = math.sqrt(1-t)

    # Angular variables
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
    """
    Generate a super fibonacci spiral of 4 dimensional points on the 3-sphere.
    See paper for formulae and justification.

    Paramters
    ---------
    n : int
        The number of  points  in the spiral
    phi: float
        A constant affecting the shape of the spiral in the first two
        coordinates
    psi: float
        A constant affecting the shape of the spiral in the second two
        coordinates

    Returns
    -------
    list of n lists of 4 floats
        The list of coordinates of the 4 dimensional super fibonacci spiral
        point on the 3 dimensional sphere
    """
    # Initialize...
    out = np.zeros((n,4))

    # ...loop over and populate.
    for i in range(n):
        s = i + 0.5
        point = super_fibonacci_point(n, s, phi=C1, psi=C2)
        out[i] = point

    return out
