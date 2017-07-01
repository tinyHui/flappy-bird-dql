from collections import deque
from random import sample

import numpy as np

from utils.rgb2train import normalize


class Memory:
    def __init__(self, memory_size, frame_per_stack):
        self.memory_size = memory_size
        self.item_count = 0
        self.memory = deque()
        self.frame_per_stack = frame_per_stack
        self.current_stack = np.zeros((2, 2, frame_per_stack))

    def remember(self, prev_state, action, reward, new_state, game_end):
        self.memory.append((prev_state, action, reward, new_state, game_end))

        if self.item_count > self.memory_size:
            self.memory.popleft()

        self.item_count += 1

    def initial_stack(self, image_data):
        frame = normalize(image_data)
        frame = np.expand_dims(frame, axis=3)
        self.current_stack = np.repeat(frame, self.frame_per_stack, axis=2)

    def stack_frame(self, image_data):
        frame = normalize(image_data)
        frame = np.expand_dims(frame, axis=3)
        self.current_stack = np.append(frame, self.current_stack[:, :, :self.frame_per_stack - 1], axis=2)

    def get_current_stack(self):
        return self.current_stack

    def get_sample_batches(self, batch_size):
        mini_batch = sample(self.memory, batch_size)

        prev_state_batch = [s[0] for s in mini_batch]
        action_batch = [s[1] for s in mini_batch]
        reward_batch = [s[2] for s in mini_batch]
        new_state_batch = [s[3] for s in mini_batch]
        game_terminate_batch = [s[4] for s in mini_batch]

        return prev_state_batch, action_batch, reward_batch, new_state_batch, game_terminate_batch

