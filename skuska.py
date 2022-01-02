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

    image, reward, done = environment.step(2.0)
    print("Reward: {} done:  + {}".format(reward, done))

    new_frame = cv2.resize(image, (840,840), interpolation=cv2.INTER_AREA)
    cv2.imshow('Bot View', new_frame)
    cv2.waitKey(1)