import logging
import sys

import numpy as np

from utils import Memory
from utils import actions
from utils.network import *

sys.path.append("game/")
import wrapped_flappy_bird as game

FORMAT = '%(stage)s | Timestamp: %(timestamp)s | Epsilon: %(epsilon)s | Action: %(action)s - %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

EPSILON = 0.1
MIN_EPSILON = 0.0001
MEMORY_SIZE = 50000
FRAME_NUM_PER_STACK = 4
OBSERVE_DURATION = 10000
ANNEAL_DURATION = 3000000
BATCH_SIZE = 32
GAMMA = 0.99
ACTIONS_CHOICE_NUMBER = 2


def get_stage_name(time_stamp):
    if time_stamp <= OBSERVE_DURATION:
        return "observe"
    elif OBSERVE_DURATION < time_stamp <= OBSERVE_DURATION + ANNEAL_DURATION:
        return "anneal + train"
    else:
        return "train"


def train(input_placeholder, output_data, sess):
    # build cost function
    action = tf.placeholder("float", [None, ACTIONS_CHOICE_NUMBER])
    y = tf.placeholder("float", [None])
    y_action = tf.reduce_sum(tf.multiply(output_data, action), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - y_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # start network
    sess.run(tf.global_variables_initializer())

    # start game
    game_state = game.GameState()

    global_timestamp = 0
    epsilon = EPSILON

    image_data, _, _ = game_state.frame_step(actions.NOTHING)

    memory = Memory(MEMORY_SIZE, FRAME_NUM_PER_STACK)
    memory.initial_stack(image_data)

    prev_state = memory.get_current_stack()

    while True:
        actions_scores = output_data.eval(feed_dict={input_placeholder: [prev_state]})[0]

        action_name, action_choice = actions.get_next_action(epsilon, actions_scores)

        image_data, reward, game_terminate = game_state.frame_step(action_choice)
        memory.stack_frame(image_data)

        new_state = memory.get_current_stack()
        memory.remember(prev_state, action_choice, reward, new_state, game_terminate)

        # anneal
        if global_timestamp > OBSERVE_DURATION and epsilon > MIN_EPSILON:
            logging.info("start anneal", extra={'stage': get_stage_name(global_timestamp),
                                                'timestamp': global_timestamp,
                                                'epsilon': epsilon,
                                                'action': action_name})
            epsilon -= float(EPSILON - MIN_EPSILON) / ANNEAL_DURATION

        # explore + train
        if global_timestamp > OBSERVE_DURATION:
            prev_state_batch, action_batch, reward_batch, new_state_batch, game_terminate_batch = memory.get_sample_batches(BATCH_SIZE)

            y_batch = []
            evaluate = output_data.eval(feed_dict={input_placeholder: new_state_batch})
            for i, game_terminate in enumerate(game_terminate_batch):
                # train target to reward
                if game_terminate:
                    y_batch.append(reward_batch[i])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * np.max(evaluate[i]))

            # gradient
            train_step.run(feed_dict={
                y: y_batch,
                action: action_batch,
                input_placeholder: new_state_batch
            })

        # update state
        prev_state = new_state
        logging.debug("finish epoch", extra={'stage': get_stage_name(global_timestamp),
                                             'timestamp': global_timestamp,
                                             'epsilon': epsilon,
                                             'action': action_name})
        global_timestamp += 1


def main():
    sess = tf.InteractiveSession()
    input_placeholder, output_data = build_network()
    train(input_placeholder, output_data, sess)


if __name__ == "__main__":
    main()
