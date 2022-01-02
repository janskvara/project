import re
from sys import dont_write_bytecode
from PIL.ImageGrab import grab
import cv2
from fastcore.basics import true
import numpy as np
import win32gui, win32ui, win32con
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pytesseract
import keyboard
import time
import pyautogui as pyautogui

class Environment():

    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'  # your path may be different
    # pytesseract.pytesseract.tesseract_cmd = r'D:/Program Files (x86)/TesseractOCR/tesseract.exe'
    _window_x_position = 0
    _window_y_position = 0
    _window_width = 800
    _window_height = 600

    _x_position = 0
    _y_position = 0

    width = 0 
    height = 0
    action_space = np.array([0.0, 1.0, 2.0], dtype=np.float32) #["up", "down", "none"]

    actual_reward = 0

    def __init__(self):

        self._browser_driver = webdriver.Chrome(ChromeDriverManager().install())
        self._browser_driver.set_window_rect(self._window_x_position, self._window_y_position, self._window_width, self._window_height)
        self._browser_driver.get("https://chromedino.com")
        self._x_position = self._window_x_position  + 50
        self._y_position = self._window_y_position + 220
        self.width =  self._window_width - 100
        self.height =  150
        self._game_over_image = cv2.imread(r'jobData\restart.png')
    
    def reset(self):
        time.sleep(0.3)
        keyboard.press_and_release('space') #game start

    def close(self):
        print("ZATVARAME :)")
    
    def step(self, action):

        if(action == 0.0):
            keyboard.press_and_release('up') #up
        
        if(action == 1.0):
            keyboard.press_and_release('down') #down
        
        if(action == 2.0):
            pass
            #do nothing

        self.actual_image = self.getScreen()
        self.actual_reward = self.getReward()
        self.done = self.getDone()

        return self.actual_image, self.actual_reward, self.done

    def getDone(self) -> bool:

        image = self.actual_image
        image_with_game_over = image[int(self.height * 0.25): int(self.height * 0.75), int(self.width*0.25) : int(self.width*0.75)]
        text = pytesseract.image_to_string(image_with_game_over)
        if "OVER" in text:
            return True
        return False

    def getReward(self) -> any:

        image = self.actual_image
        image_with_number = image[0: int(self.height * 0.3), int(self.width*0.8) : self.width]
        text = pytesseract.image_to_string(image_with_number)
        try:
            self.actual_reward = int(text[1:])
            #print(text[1:])
        except:
            self.actual_reward = self.actual_reward

        return self.actual_reward

    def getWhiteBlackScreen(self) -> any:

        image = cv2.cvtColor(self.actual_image, cv2.COLOR_BGR2GRAY)
        (thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return image

    def getScreen(self) -> any:
        
        hwin = win32gui.GetDesktopWindow()
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, self.width, self.height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (self.width, self.height), srcdc, (self._x_position,  self._y_position), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.height, self.width, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)