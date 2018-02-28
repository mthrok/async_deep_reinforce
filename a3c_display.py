import numpy as np
import tensorflow as tf

import constants
from game_state import GameState
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork


def choose_action(pi_values):
    return np.random.choice(range(len(pi_values)), p=pi_values)


def _main():
    # use CPU for display tool
    device = "/cpu:0"

    Network = GameACLSTMNetwork if constants.USE_LSTM else GameACFFNetwork
    global_network = Network(constants.ACTION_SIZE, -1, device)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(constants.CHECKPOINT_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("checkpoint loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old checkpoint")

    game_state = GameState(0, display=True, no_op_max=0)

    while True:
        pi_values = global_network.run_policy(sess, game_state.s_t)

        game_state.process(choose_action(pi_values))
        if game_state.terminal:
            game_state.reset()
        else:
            game_state.update()


if __name__ == '__main__':
    _main()
