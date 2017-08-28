from train_ac import train
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', default=15)
ap.add_argument('-s', '--steps', default=4)
ap.add_argument('-lr', '--learningrate', default=1e-4)
ap.add_argument('-w', '--wait', default=0.05)
args = vars(ap.parse_args())

train(int(args['epochs']), int(args['steps']), float(args['wait']), float(args['learningrate']))
