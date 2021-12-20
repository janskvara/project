from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import cv2
import os
import keyboard
from grabScreen import grab_screen
import uuid

_x_position = 0
_y_position = 0
width = 800
height = 600

driver = webdriver.Chrome(ChromeDriverManager().install())
driver.set_window_rect(_x_position, _y_position, width, height)

# get chromedinoS
driver.get("https://chromedino.com")

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

# Collets images until you press e
while not keyboard.is_pressed("e"):
    count += 1
    image = grab_screen(region = (_x_position + 100, _y_position + 300, width, 200))
    #Covert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert to black or white pixel
    (thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Bot View', image)
    cv2.waitKey(1)

    if keyboard.is_pressed(' '):
        cv2.imwrite(f"{_up_path}\{uuid.uuid4()}.jpg", image)
        continue

    if keyboard.is_pressed('down'):
        cv2.imwrite(f"{_down_path}\{uuid.uuid4()}.jpg", image)
        continue

    if count%20 == 0:
        cv2.imwrite(f"{_nothing_path}\{uuid.uuid4()}.jpg", image)