import cv2
import os
import keyboard
from environment_flappy import Environment
import uuid

# set window position
count = 0

_up_path = r'.\data_flappy\df_up84'
if not os.path.exists(_up_path):
    os.makedirs(_up_path)

_nothing_path = r'.\data_flappy\df_nothing84'
if not os.path.exists(_nothing_path):
    os.makedirs(_nothing_path)

environment = Environment()

# Collets images until you press e
while not keyboard.is_pressed("e"):
    count += 1
    environment.getScreen()
    image = environment.getWhiteBlackScreen()

    new_frame = cv2.resize(image, (840,840), interpolation=cv2.INTER_AREA)
    cv2.imshow('Bot View', new_frame)
    cv2.waitKey(1)

    if keyboard.is_pressed(' ') or keyboard.is_pressed('up'):
        cv2.imwrite(f"{_up_path}\{uuid.uuid4()}.jpg", image)
        continue

    cv2.imwrite(f"{_nothing_path}\{uuid.uuid4()}.jpg", image)