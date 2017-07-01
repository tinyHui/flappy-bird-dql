from random import random, choice

import numpy as np

NOTHING = np.array([1., 0.])
FLAP = np.array([0., 1.])


def get_next_action(epsilon, t):
    if random() <= epsilon:
        return choice((("random NOTHING", NOTHING), ("random FLAP", FLAP)))
    else:
        # choice the action with best score
        action_index = np.argmax(t)
        return (("NOTHING", NOTHING), ("FLAP", FLAP))[action_index]
