"""
translators.py: contains definitions of dictionaries used to translate classes from different experiments into zero-indexed integers.
"""

# Shape
SHAPE_DICT = {
    -1: 0,
    1: 1,
}

# Gravity Bias
GRAVITY_DICT = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
}

# Simple Swap
SWAP_DICT = {
    "left": 0,
    "middle": 1,
    "right": 2,
}

# Rotation
ROTATION_DICT = {
    "left": 0,
    "middle": 1,
    "right": 2,
}

# Addition Numbers
ADDITION_DICT = {
    "left": 0,
    "equal": 1,
    "right": 2,
}

# Relative Numbers
RELATIVE_DICT = {
    "left": 0,
    "right": 1,
}

expt_dicts = {
    "shape": SHAPE_DICT,
    "gravity": GRAVITY_DICT,
    "swap": SWAP_DICT,
    "rotation": ROTATION_DICT,
    "addition": ADDITION_DICT,
    "relative": RELATIVE_DICT,
}
