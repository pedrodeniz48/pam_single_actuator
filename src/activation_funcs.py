# Activation functions
import math

def slow(t):
    return (
        0.055
        - 0.045 * math.cos(
            2 * math.pi * 0.25 * t
        )
    )

def step(t):
    if t < 1.0:
        return 0.000001
    else:
        return 0.10

def sine(t, f=1):
    return (
        0.055
        - 0.045
        * math.cos(
            2 * math.pi * f * t
        )
    )
