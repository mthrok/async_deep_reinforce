# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib.pyplot as plt

import constants
from game_ac_network import GameACFFNetwork, GameACLSTMNetwork


def _main():
    # use CPU for weight visualize tool
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

    W_conv1 = sess.run(global_network.W_conv1)

    # show graph of W_conv1
    fig, axes = plt.subplots(
        4, 16, figsize=(12, 6),
        subplot_kw={'xticks': [], 'yticks': []})
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for ax, i in zip(axes.flat, range(4*16)):
        inch = i//16
        outch = i % 16
        img = W_conv1[:, :, inch, outch]
        ax.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
        ax.set_title(str(inch) + "," + str(outch))
    plt.show()


if __name__ == '__main__':
    _main()
