from grabScreen import grab_screen
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import cv2
import keyboard
from fastai.vision.all import *

def label_func(x): return x.parent.name

def load(path):
    learner = load_learner(path)
    return(learner)

def decide(learner):
    # set window position
    _x_position = 0
    _y_position = 0
    width = 800

    image = grab_screen(region = (_x_position + 100, _y_position + 300, width, 200))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert to binary colors
    (_, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Bot View', image)
    cv2.waitKey(1)

    # Decide about next step
    result = learner.predict(image)
    action = result[0]

    return action

def run():
    # set window position
    _x_position = 0
    _y_position = 0
    width = 800
    height = 600
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.set_window_rect(_x_position, _y_position, width, height)

    # get chromedinoS
    driver.get("https://chromedino.com")

    while not keyboard.is_pressed("e"):
        action = decide(learner)

        if action == 1:
            keyboard.release()
            keyboard.press('down')
        elif action == 2:
            keyboard.press('up')
        else:
            continue

if __name__ == '__main__':
    learner = load('model.pkl')
    run()


