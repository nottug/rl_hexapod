# Hexapod robot

This is a project I've been working on for the past year that uses deep reinforcment learning (Actor Critic) to teach the hexapod how to walk. The goal is for the hexapod to follow a person using visual recognition. It runs off PyTorch and uses [YOLO](https://github.com/longcw/yolo2-pytorch) for the object recognition. Some of the code for training was used from an [A3C implementation](https://github.com/ikostrikov/pytorch-a3c).

Everything is running of the Jetson TX1 and the hexapod was built from scratch. The .STL files were 3D printed and the .DXF files were carbon fiber and cut using a CNC mill.
