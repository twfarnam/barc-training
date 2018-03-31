#! /usr/bin/env python3

import argparse
from src.database import categories
from src.mobilenets import train_mobilenets
from src.inception import train_inception

parser = argparse.ArgumentParser(
    description='Train a neural network from Barc images'
)

parser.add_argument(
    '--count',
    action='store_true',
    help='print a list of image counts by categories and exit',
)

parser.add_argument(
    '--architecture',
    action='store',
    dest='architecture',
    nargs=1,
    type=str,
    default=[ 'mobilenets' ],
    choices=[ 'mobilenets', 'inception' ],
    help='which architecture to train, default is mobilenets',
)


parser.add_argument(
    '--epochs',
    action='store',
    dest='epochs',
    nargs=1,
    type=int,
    default=[ 20 ],
    help='number of epochs to train',
)

args = parser.parse_args()

if args.count:
    ids, labels, counts = categories()
    for i, label in enumerate(labels):
        print("{:<35}{:>4}".format(label, counts[i]))
elif args.architecture[0] == 'inception':
    train_inception(epochs=args.epochs[0])
elif args.architecture[0] == 'mobilenets':
    train_mobilenets(epochs=args.epochs[0])

