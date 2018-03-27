#! /usr/bin/env python

import argparse
from src.database import count_by_category
from src.mobilenets import train_mobilenets
from src.inception import train_inception

parser = argparse.ArgumentParser(description='Train a neural network from Barc images')


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
    default=[ 1 ],
    help='number of epochs to train',
)

parser.add_argument(
    '--log-dir',
    action='store',
    dest='log_dir',
    nargs=1,
    type=str,
    default=[ './log' ],
    help='output directory for TensorBoard, default ./log',
)

args = parser.parse_args()

if args.count:
    # print(count_by_category())
    for category, n in count_by_category():
        print("{:>15}{:>15}".format(category, n))
elif args.architecture[0] == 'inception':
    train_inception(epochs=args.epochs[0], log_dir=args.log_dir[0])
elif args.architecture[0] == 'mobilenets':
    train_mobilenets(epochs=args.epochs[0], log_dir=args.log_dir[0])

