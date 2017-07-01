from collections import deque

import numpy as np

from utils.rgb2train import normalize


class Memory:
    def __init__(self, memory_size, frame_per_stack):
        self.memory_size = memory_size
        self.item_count = 0
        self.memory = deque()
        self.current_stack = np.zeros((2, 2, 4))

    def remember(self, prev_state, action, reward, new_state, game_end):
        self.memory.append((prev_state, action, reward, new_state, game_end))

        if self.item_count > self.memory_size:
            self.memory.popleft()

        self.item_count += 1

    def initial_stack(self, image_data):
        frame = normalize(image_data)
        self.current_stack = np.stack((frame, frame, frame, frame), axis=2)

    def stack_frame(self, image_data):
        frame = normalize(image_data)
        frame = np.expand_dims(frame, axis=3)
        self.current_stack = np.append(frame, self.current_stack[:, :, :3], axis=2)

    def get_current_stack(self):
        return self.current_stack
