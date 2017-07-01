import logging
import sys

from utils import Memory
from utils import actions

sys.path.append("game/")
import wrapped_flappy_bird as game

FORMAT = '%(levelname)s | Timestamp: %(timestamp)s | Flap Chance: %(flap_chance)s | Action: %(action)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)


INITIAL_CHANCE_OF_FLAP = 0.05
MIN_CHANCE_OF_FLAP = 0.00005
MEMORY_SIZE = 50000
FRAME_NUM_PER_STACK = 4


def main():
    game_state = game.GameState()

    global_timestamp = 0
    chance_of_flap = INITIAL_CHANCE_OF_FLAP

    image_data, reward, game_end = game_state.frame_step(actions.NOTHING)

    memory = Memory(MEMORY_SIZE, FRAME_NUM_PER_STACK)
    memory.initial_stack(image_data)

    prev_state = memory.get_current_stack()

    while True:
        action_name, action = actions.get_random_action(chance_of_flap)

        image_data, reward, game_terminate = game_state.frame_step(action)
        memory.stack_frame(image_data)

        new_state = memory.get_current_stack()
        memory.remember(prev_state, action, reward, new_state, game_terminate)

        logging.debug('take action', extra={'timestamp': global_timestamp,
                                            'flap_chance': chance_of_flap,
                                            'action': action_name})
        global_timestamp += 1


if __name__ == "__main__":
    main()
