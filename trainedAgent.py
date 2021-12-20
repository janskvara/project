from torch._C import wait
from grabScreen import grab_screen
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from fastai.vision.all import *
import keyboard
import time
import pyautogui as pyautogui
import cv2

def label_func(x): return x.parent.name

def scan_for_restart(): # scans for the game restart
    result = pyautogui.locateOnScreen(r'dataset\restart.png', grayscale=True, region=(100, 150, 400, 350), confidence=0.9) 
    if result != None: # restart button located
        time.sleep(1)
        print('RESTART')
        keyboard.press_and_release('space')

def load(path):
    learner = load_learner(path)
    print('Model loaded.')
    return(learner)

def decide(learner):
    # set window position
    _x_position = 0
    _y_position = 0
    width = 800

    image = grab_screen(region = (_x_position + 100, _y_position + 300, width, 200))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # convert to binary colors
    (_, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('Bot View', image)
    cv2.waitKey(1)

    # Decide about next step
    result = learner.predict(image)
    action = result[0]

    return action #ddown / dnothing / dup

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

    prevKey = None
    # decide about next moves until user presses 'e' key
    while not keyboard.is_pressed("e"):
        scan_for_restart()
        action = decide(learner)

        # transfer from directory name to key name
        if action == 'ddown':
            keyToPress = 'down'
        elif action == 'dup':
            keyToPress = 'up'
        else:
            continue # do nothing

        # key hold or switch of the key that is being held
        if keyToPress == prevKey:
            continue
        elif keyToPress != prevKey:
            if prevKey != None:
                keyboard.release(prevKey)
            keyboard.press(keyToPress)
            print('Input: ' + keyToPress)
        prevKey = keyToPress

if __name__ == '__main__':
    learner = load('dataset\model.pkl')
    run()

