#!/usr/bin/env python
import argparse

import sys
import numpy as np
import keras
import keras.backend as K
from keras import Sequential
from keras.callbacks import Callback
from keras.engine.saving import load_model
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2

RANDOM_MOVES_UNTIL = 100 * 1000
UPDATE_AFTER_MOVES = 100 * 1000

L2_REGULARISATION = 0.0002
DECAY = 0.001
LR = 0.005
STOP_ON_PERCENT = 5.0
BOARD_SIZE = 3
WIN_SIZE = 3
LAYER1_SIZE = 64
LAYER2_SIZE = 32
EPOCHS = 2
FILENAME = 'tictactoe.h5'
HALF_LEARNING_RATE_PER_M = 4.0


class LRCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        print K.eval(lr_with_decay)


def lr_schedule(epoch, lr):
    mill = int(len(board_weights) / 1000 / 1000 / HALF_LEARNING_RATE_PER_M)
    return LR / (1 << mill)


def create_model():
    model = Sequential()
    model.add(Conv2D(LAYER1_SIZE, activation="relu", kernel_size=(3, 3),
                     input_shape=(2, BOARD_SIZE, BOARD_SIZE),
                     data_format="channels_first",
                     kernel_regularizer=l2(L2_REGULARISATION),
                     padding='same'))
    model.add(Conv2D(LAYER1_SIZE, activation="relu", kernel_size=(3, 3),
                     data_format="channels_first",
                     kernel_regularizer=l2(L2_REGULARISATION),
                     padding='same'))
    model.add(MaxPooling2D((2, 2), data_format="channels_first"))
    model.add(Conv2D(LAYER1_SIZE * 2, activation="relu", kernel_size=(3, 3),
                     data_format="channels_first",
                     kernel_regularizer=l2(L2_REGULARISATION),
                     padding='same'))
    model.add(Conv2D(LAYER1_SIZE * 2, activation="relu", kernel_size=(3, 3),
                     data_format="channels_first",
                     kernel_regularizer=l2(L2_REGULARISATION),
                     padding='same'))
    model.add(MaxPooling2D((2, 2), data_format="channels_first"))
    model.add(Flatten())
    model.add(Dense(LAYER2_SIZE, activation='relu', kernel_regularizer=l2(L2_REGULARISATION)))
    model.add(Dense(1, activation='tanh'))

    optimizer = Adam(decay=DECAY, lr=LR)
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy', 'mae'])
    model.summary()

    return model


def train(model, boards, current_wins):
    print "Training on %d moves" % len(boards)
    if not boards:
        return

    lr_print_callback = LRCallback()
    log_callback = keras.callbacks.TensorBoard()
    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
    save_callback = keras.callbacks.ModelCheckpoint(FILENAME)

    history = model.fit(np.array(boards), np.array(current_wins),
                        epochs=EPOCHS,
                        validation_split=0.1,
                        verbose=0,
                        batch_size=256,
                        callbacks=[lr_print_callback, log_callback, lr_callback, save_callback])

    history_dict = history.history
    print " Acc %s \n MAE %s \n VAL MEA %s" % \
          (history_dict['acc'],
           history_dict['mean_absolute_error'],
           history_dict['val_mean_absolute_error'])


def find_best_move(model, board, player):
    best_prob_to_win = None
    best_x = None
    best_y = None
    if player == 0:
        test_board = np.flip(board, axis=0)
    else:
        test_board = board.copy()

    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if not test_board[0, x, y] and not test_board[1, x, y]:
                test_board[1, x, y] = 1
                prob_to_win = model.predict(np.array([test_board]), batch_size=1, verbose=0)[0][0]
                test_board[1, x, y] = 0
                if best_prob_to_win is None or prob_to_win < best_prob_to_win:
                    best_x = x
                    best_y = y
                    best_prob_to_win = prob_to_win

    return best_x, best_y, best_prob_to_win


def get_rotated(board):
    ret = []
    for rot in range(1, 4):
        rotated_board = np.array((np.rot90(board[0], k=rot), np.rot90(board[1], k=rot))).copy()
        ret.append(rotated_board)
    return ret


def get_flipped(board):
    ret = []
    for axis in range(1, 3):
        flipped_board = np.flip(board, axis=axis).copy()
        ret.append(flipped_board)
    return ret


def update_board_weight(board_weights, board, winner):
    if not board.flags.contiguous:
        board = board.copy()
    board.flags.writeable = False
    key = hash(board.data)
    if key not in board_weights:
        board_weights[key] = [board, np.zeros(3)]

    if winner == 0:
        board_weights[key][1] += np.array([1, 0, 0])
    elif winner == 1:
        board_weights[key][1] += np.array([0, 0, 1])
    elif winner == 0.5:
        board_weights[key][1] += np.array([0, 1, 0])

    return key


def get_training_data(board_weights, keys):
    boards = []
    weights = []
    only_1_sample = 0
    for key in keys:
        board_weight = board_weights[key]
        board = board_weight[0]
        weight_arr = board_weight[1]
        total = np.sum(weight_arr)
        weight = (weight_arr[0] * 1 + weight_arr[1] * 0 + weight_arr[2] * -1) / total

        if total == 1:
            only_1_sample += 1
        else:
            pass

        boards.append(board)
        weights.append(weight)

    only_1_sample_percent = 100. * only_1_sample / float(len(keys))
    print "%d / %d boards (%.1f %%) had only 1 sample" % (only_1_sample, len(keys), only_1_sample_percent)
    return boards, weights


def store_boards(board_weights, current_boards, updated_board_keys, winner):
    for i in xrange(len(current_boards)):
        board = current_boards[i]
        if i % 2 == 0:
            score = 1 - winner
            board = np.flip(board, axis=0).copy()
        else:
            score = winner

        key = update_board_weight(board_weights, board, score)
        updated_board_keys.append(key)

        for transformed_board in get_flipped(board):
            key = update_board_weight(board_weights, transformed_board, score)
            updated_board_keys.append(key)

        for transformed_board in get_rotated(board):
            key = update_board_weight(board_weights, transformed_board, score)
            updated_board_keys.append(key)


board_weights = {}
updated_board_keys = []


def game_ended(model, winner, current_boards):
    global board_weights, updated_board_keys

    store_boards(board_weights, current_boards, updated_board_keys, winner)

    if len(updated_board_keys) >= UPDATE_AFTER_MOVES:
        boards, weights = get_training_data(board_weights, updated_board_keys)
        train(model, boards, weights)

        test_model(model)
        updated_board_keys = []


def get_valid_moves(board):
    valid_moves = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if not board[0, x, y] and not board[1, x, y]:
                valid_moves.append((x, y))
    return valid_moves


def get_random_move(board):
    valid_moves = get_valid_moves(board)
    return valid_moves[np.random.randint(len(valid_moves))]


def has_won(board, player):
    for start_x in range(BOARD_SIZE - WIN_SIZE + 1):
        for start_y in range(BOARD_SIZE - WIN_SIZE + 1):
            if has_won_at(start_x, start_y, board, player):
                return True

    return False


def has_won_at(start_x, start_y, board, player):
    p = player

    won = np.zeros(4)

    for x in range(WIN_SIZE):
        won[0] += board[p, start_x + x, start_y + x]
        won[1] += board[p, start_x + x, start_y + WIN_SIZE - 1 - x]
        for y in range(WIN_SIZE):
            won[2] += board[p, start_x + x, start_y + y]
            won[3] += board[p, start_x + y, start_y + x]

        if won[2] == WIN_SIZE or won[3] == WIN_SIZE:
            return True
        won[2] = won[3] = 0

    return won[0] == WIN_SIZE or won[1] == WIN_SIZE


def is_winner_move(player, start_x, start_y, board):
    if not board[player, start_x, start_y]:
        return False

    counts = np.ones((3, 3))
    moves = ([-1, 0], [-1, 1], [0, 1], [1, 1],
             [-1, -1], [0, -1], [1, -1], [1, 0])

    for i in range(1, WIN_SIZE):
        for move in moves:
            count_x, count_y = 1 + move[0], 1 + move[1]
            if counts[count_x, count_y] != i:
                continue

            x, y = start_x + i * move[0], start_y + i * move[1]
            if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
                continue

            if board[player, x, y]:
                counts[count_x, count_y] += 1

    for i in range(len(moves) / 2):
        move = moves[i]
        countermove = moves[len(moves) - 1 - i]
        x, y = 1 + move[0], 1 + move[1]
        cx, cy = 1 + countermove[0], 1 + countermove[1]
        total = counts[x, y] + counts[cx, cy] - 1
        if total >= WIN_SIZE:
            return True

    return False


def is_board_full(board):
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            if not board[0, x, y] and not board[1, x, y]:
                return False

    return True


def train_by_random(model):
    random_won = 0
    draw = 0
    batch_size = 500
    total_moves = 0

    for i in xrange(1000 * 1000):
        randomized_player = np.random.randint(2)

        if total_moves < RANDOM_MOVES_UNTIL:
            randomized_player = 2

        winner, boards = play_against_self(model, randomized_player)
        total_moves += len(boards)

        randomized_player_won = randomized_player == 2 or randomized_player == winner
        draw += winner == 0.5

        if randomized_player_won:
            random_won += 1

        game_ended(model, winner, boards)

        if i and i % batch_size == 0:
            random_win_percent = 100. * random_won / batch_size
            draw_percent = 100. * draw / batch_size
            print "%d games, %d moves so far. Random wins %.1f %%, " \
                  "Draw %.1f %% of the time." \
                  % (i, total_moves, random_win_percent, draw_percent)

            random_won = 0
            draw = 0

            if random_win_percent <= STOP_ON_PERCENT:
                model.save(FILENAME)
                sys.exit('Model trained (<%.2f%% error), quitting' %
                         STOP_ON_PERCENT)


def test_model(model):
    win = 0
    lose = 0
    draw = 0
    winner, _ = play_against_self(model, randomized_player=None,
                                  printgame=True)

    if winner == 0.5:
        draw += 1
    if winner == 1:
        lose += 1
    if winner == 0:
        win += 1

    print "Model against itself win %d draw %d lose %d" % (win, draw, lose)
    return float(win)


def board_to_str(board):
    board_string = ''
    for x in range(BOARD_SIZE):
        line = '|'
        for y in range(BOARD_SIZE):
            if board[0, x, y]:
                line += 'X'
            elif board[1, x, y]:
                line += '0'
            else:
                line += ' '
        line += '|'
        board_string += line
        board_string += '\n'
    return board_string.rstrip()


def calculate_random_chance(certainty):
    return -0.25 * certainty + .7


def play_against_self(model, randomized_player=None, printgame=False):
    boards = []
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE))
    player = 0

    if printgame:
        print

    while True:
        if randomized_player == 2:
            x, y = get_random_move(board)
            value = 'rnd'
        else:
            x, y, value = find_best_move(model, board, player)

            if randomized_player == player:
                # Find balance between exploit / explore
                rnd_chance = calculate_random_chance(value)
                if np.random.random() > rnd_chance:
                    x, y = get_random_move(board)
                    value = 'rnd'

        board[player, x, y] = 1

        if printgame:
            print board_to_str(board)
            print "Player %d (%d,%d) %s" % (player, x, y, value)

        boards.append(board.copy())

        if is_winner_move(player, x, y, board):
            return player, boards

        if is_board_full(board):
            return 0.5, boards

        if player == 0:
            player = 1
        else:
            player = 0


def play_against_model(model):
    board = np.zeros((2, BOARD_SIZE, BOARD_SIZE))

    def check_board(x, y):
        if is_winner_move(0, x, y, board):
            print board_to_str(board)
            sys.exit('You won')

        if is_winner_move(1, x, y, board):
            print board_to_str(board)
            sys.exit('You lose')

        if is_board_full(board):
            print board_to_str(board)
            sys.exit('Draw')

    while True:
        print board_to_str(board)
        x, y = input('Your move? (x, y) ')

        board[0, x, y] = 1
        check_board(x, y)

        x, y, _ = find_best_move(model, board, 1)
        board[1, x, y] = 1
        check_board(x, y)


def eval_boards(boards_str, model):
    for board_str in boards_str.split('\n\n'):
        board, score = eval_board(board_str, model)
        print board_to_str(board)
        print score


def eval_board(board_str, model):
    _, _, board_size, board_size = model.layers[0].input_shape
    board = parse_board(board_size, board_str)
    return board, model.predict(np.array([board]), batch_size=1, verbose=0)[0][0]


def parse_board(board_size, board_str):
    board = np.zeros((2, board_size, board_size))

    board_rows = board_str.split('\n')
    for i in xrange(len(board_rows)):
        row = board_rows[i]
        for j in xrange(len(row)):
            player = row[j]
            if player == ' ':
                continue
            if player == 'X':
                board[0][i][j] = 1.
            if player == 'O':
                board[1][i][j] = 1.

    return board


def parse_command_line():
    parser = argparse.ArgumentParser(
        description='Train/Play TicTacToe',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("cmd", help="Train a new model, play against model, "
                                    "or evaluate boards",
                        choices=['train', 'play', 'eval'])

    parser.add_argument('-f', '--filename',
                        help='Model file to load/save',
                        default=FILENAME)

    parser.add_argument('--decay',
                        help='Training decay',
                        type=float,
                        default=DECAY)

    parser.add_argument('--lr',
                        help='Learning rate',
                        type=float,
                        default=LR)

    parser.add_argument('--stopon',
                        help='Stop training when random only beats model by %%',
                        type=float,
                        default=STOP_ON_PERCENT)

    parser.add_argument('--boardsize',
                        help='The size of the board (N x N)',
                        type=int,
                        default=BOARD_SIZE)

    parser.add_argument('--winsize',
                        help='The number of items in a row needed to win',
                        type=int,
                        default=WIN_SIZE)

    parser.add_argument('--layer1size',
                        help='The number perceptrons on 1st layer',
                        type=int,
                        default=LAYER1_SIZE)

    parser.add_argument('--layer2size',
                        help='The number perceptrons on 2nd layer',
                        type=int,
                        default=LAYER2_SIZE)

    parser.add_argument('--randomuntil',
                        help='Random moves in the first N games',
                        type=int,
                        default=RANDOM_MOVES_UNTIL)

    parser.add_argument('--updatebatchsize',
                        help='Update the model after N moves',
                        type=int,
                        default=UPDATE_AFTER_MOVES)

    parser.add_argument('--epochs',
                        help='Epochs per training batch',
                        type=int,
                        default=EPOCHS)

    parser.add_argument('--halflr',
                        help='Half learning rate every M million of moves',
                        type=float,
                        default=HALF_LEARNING_RATE_PER_M)

    args = parser.parse_args()

    return args


def main(args):
    if args.cmd == 'train':
        model = create_model()
        train_by_random(model)

    if args.cmd == 'play':
        model = load_model(args.filename)
        model.summary()
        play_against_model(model)

    if args.cmd == 'eval':
        model = load_model(args.filename)
        model.summary()
        boards = ""
        for line in sys.stdin:
            boards += line
        eval_boards(boards, model)


if __name__ == "__main__":
    ARGS = parse_command_line()
    DECAY = ARGS.decay
    LR = ARGS.lr
    STOP_ON_PERCENT = ARGS.stopon
    BOARD_SIZE = ARGS.boardsize
    WIN_SIZE = ARGS.winsize
    LAYER1_SIZE = ARGS.layer1size
    LAYER2_SIZE = ARGS.layer2size
    RANDOM_MOVES_UNTIL = ARGS.randomuntil
    UPDATE_AFTER_MOVES = ARGS.updatebatchsize
    EPOCHS = ARGS.epochs
    FILENAME = ARGS.filename
    HALF_LEARNING_RATE_PER_M = ARGS.halflr

    main(ARGS)
