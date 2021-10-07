import os
import numpy as np
from ai2thor.controller import Controller
import random
import cv2
import random
from tqdm import tqdm
from math import erf, sqrt


class Experiment(Controller):
    def __init__(self, controller_args, seed):
        random.seed(seed)
        np.random.seed(seed)
        super().__init__(**controller_args)