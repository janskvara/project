from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import cv2
import os
import keyboard
from environment import Environment
import uuid

# set window position
count = 0

_up_path = r'.\dataset\dup' 
if not os.path.exists(_up_path):
    os.makedirs(_up_path)

_down_path = r'.\dataset\ddown' 
if not os.path.exists(_down_path):
    os.makedirs(_down_path)

_nothing_path = r'.\dataset\dnothing' 
if not os.path.exists(_nothing_path):
    os.makedirs(_nothing_path)

environment = Environment()

# Collets images until you press e
while not keyboard.is_pressed("e"):
    count += 1
    image = environment.getWhiteBlackScreen()

    cv2.imshow('Bot View', image)
    cv2.waitKey(1)

    if keyboard.is_pressed(' ') or keyboard.is_pressed('up'):
        cv2.imwrite(f"{_up_path}\{uuid.uuid4()}.jpg", image)
        continue

    elif keyboard.is_pressed('down'):
        cv2.imwrite(f"{_down_path}\{uuid.uuid4()}.jpg", image)
        continue
    
    else:
        cv2.imwrite(f"{_nothing_path}\{uuid.uuid4()}.jpg", image)