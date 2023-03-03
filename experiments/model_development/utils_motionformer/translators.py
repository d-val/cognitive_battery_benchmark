"""
translators.py: contains definitions of functions used to translate classes from different expts into zero-indexed integers.
"""

# Translator for the Shape experiment
def SHAPE(label):
    return {
        -1: 0,
        1: 1
    }[label]

# Translator for the Gravity Bias experiment
def GRAVITY(label):
    return {
        0: 0,
        1: 1,
        2: 2
    }[label]

# Translator for the Simple Swap experiment
def SWAP(label):
    return {
        "left": 0,
        "middle": 1,
        "right": 2
    }[label]

expts = {
    "shape": SHAPE,
    "gravity": GRAVITY,
    "swap": SWAP
    }
