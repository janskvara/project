import cv2
import os
import keyboard
from environment import Environment
import uuid

# set window position
count = 0

environment = Environment()
environment.reset()
# Collets images until you press e
done = False
while not done:
    count += 1

    image, reward, done = environment._step(2.0)
    print("Reward: {} done:  + {}".format(reward, done))

    #new_frame = cv2.resize(image, (840,840))
    cv2.imshow('Bot View', image)
    cv2.waitKey(1)