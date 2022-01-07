# The autonomous gamer

a repository for the semestral project in the course 
**Advanced Artificial Intelligence for Games** on TEK SDU Odense

The project is focused on using AI to play simple games via screen-capture technology, drawing data directly from the screen and in real time. Main focus is on the *chrome dino* game, which, albeit simple, still makes the AI training hard by implementing a lot of randomness, each run having a different, generated track.

![chrome dino](https://miro.medium.com/max/1200/1*82D2cg8Gpe9CVISaph6RPg.gif)

The technologies used are **convolutional neural networks** (*pytorch* backend), specifically ResNet and SqueezeNet and Deep Reinforcement Learning (see the [dqn branch](https://github.com/janskvara/project/tree/dqn)).

# Prerequisites

The list of needed python libraries can be found in the file `requirements.txt`

# List of usable scripts:

 - `dataCollect.py` - collects data for the dataset, using screen capture technology
 - `dataCollect_flappy.py` - collects data for the flappy bird dataset
 - `environment.py` - mother class, containing utility functions for other scripts
 - `trainFromExisting.py` - main training script for CNNs
 - `trainedAgent.py` - script for testing pretrained models on real-time version of the game

In the DQN branch:

 - `main.py` - runnable script, contains a CNN and an advanced DQN training script
 - `main_random.py` - random agent testing script - used for benchmarking other models
 - `simple_main.py` - basic DQN training script
 - `myDQN.py` - contains utilities for DQN algorithms
 - `random_agent.py` - contains utilities for the random agent
 - `replay.py` - contains the algorithms simpleReplayBuffer and prioritizedExperienceReplayBuffer

# Models

Pretrained models can be found in the *models* folder. 
The name of the model contains it's specifications:
**model_ architecture type _ number of layers _ number of epochs _  training dataset size**

For example `model_resnet-18_5_80k.pkl` is a ResNet with 18 layers, trained on 80 000 images for 5 training cycles (epochs).
