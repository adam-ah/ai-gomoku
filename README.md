# Keras zero knowledge AI for gomoku and tic-tac-toe

## Prerequisites

Keras with a valid backend (eg, tensorflow)
 
    pip install tensorflow
    pip install keras

To install optimised Tensorflow backend, go to https://github.com/lakshayg/tensorflow-build


## Training 

The system will start playing randomly against itself with the predefined parameters. 

For board size 6x6 with 5 in a row for winning, stopping after random only wins 2% of the games

    ./ai_gomoku.py train --boardsize 6 --winsize 5 --stopon 2 --decay 0 --lr 0.001 --layer1size 128 --layer2size 128 --epochs 2 --randomuntil 0 --updatebatchsize 10000

<i>Training time ~5 hours without GPU.</i>

To watch the progress, start

    tensorboard --logdir=logs/
    
and head to http://localhost:6006 

## Playing agianst the model

The player starts with X. Input the moves in the format of X, Y where X is the row, Y is the column, starting from zero.

    ./ai_gomoku.py play --boardsize 6 --winsize 5 --filename gomoku_6_5.h5 
    
Eg,

    |      |
    |      |
    |      |
    |      |
    |      |
    |      |
    Your move? (x, y) 3,3
    |      |
    |      |
    |   0  |
    |   X  |
    |      |
    |      |
    ...
    
## Evaluating boards

Copy-paste boards to STDIN to evaluate it's score based on a model. Boards are separated with two empty lines. End with a CMD+D.

Sample boards are provided in eval_boards.txt

To indicate empty rows add a single space in that line.

    ./ai_gomoku.py eval --boardsize 6 --winsize 5 --filename gomoku_6_5.h5 