from environment import Environment
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from fastai.vision.all import *
import keyboard
import time
import pyautogui as pyautogui
import cv2

def label_func(x): return x.parent.name

def scan_for_restart(): # scans for the game restart
    result = pyautogui.locateOnScreen(r'jobData\restart.png', grayscale=True, region=(100, 150, 400, 350), confidence=0.9) 
    if result != None: # restart button located
        time.sleep(1)
        print('RESTART')
        keyboard.press_and_release('space')

def load(path): # loads trained model
    learner = load_learner(path)
    print('Model loaded.')
    return(learner)

def decide(learner): 
    image = environment.getWhiteBlackScreen()
    cv2.imshow('Bot View', image) 
    cv2.waitKey(1)

    # Decide about next step
    result = learner.predict(image)
    action = result[0]

    return action #ddown / dnothing / dup

def run():
    prevKey = None
    keyboard.press_and_release('space') #game start
    
    # decide about next moves until user presses 'e' key
    while not keyboard.is_pressed("e"):

        #scan_for_restart()
        action = decide(learner)

        # transfer from directory name to key name
        if action == 'ddown':
            keyToPress = 'down'
        elif action == 'dup':
            keyToPress = 'up'
        else:
            continue # do nothing

        # key hold or switch of the key that is being held
        keyboard.press_and_release(keyToPress)
        '''if keyToPress == prevKey:
            continue
        elif keyToPress != prevKey:
            if prevKey != None:
                keyboard.release(prevKey)
            keyboard.press(keyToPress)
            print('Input: ' + keyToPress)
        prevKey = keyToPress'''

if __name__ == '__main__':
    
    # for models trained on colab
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    
    environment = Environment()
    learner = load('jobData\model_resn-6.pkl')
    run()