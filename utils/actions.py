from random import random

import numpy as np

NOTHING = np.array([1, 0])
FLAP = np.array([0, 1])


def get_random_action(flap_chance):
    if random() <= flap_chance:
        return 'FLAP', FLAP
    else:
        return 'NOTHING', NOTHING

