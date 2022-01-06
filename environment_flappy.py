import cv2
import numpy as np
import win32gui, win32ui, win32con
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import keyboard
import time
KMP_DUPLICATE_LIB_OK='TRUE'

class Environment():
    
    _window_x_position = 0
    _window_y_position = 0
    _window_width = 800
    _window_height = 1000

    _x_position = 0
    _y_position = 0

    width = 0 
    height = 0
    action_space = np.array([0.0, 1.0, 2.0], dtype=np.float32) #["up", "down", "none"]

    actual_reward = 0

    def __init__(self):

        self._browser_driver = webdriver.Chrome(ChromeDriverManager().install())
        self._browser_driver.set_window_rect(self._window_x_position, self._window_y_position, self._window_width, self._window_height)
        self._browser_driver.get("https://flappybird.io/")
        self._x_position = self._window_x_position  + 50
        self._y_position = self._window_y_position + 220
        self.width =  self._window_width - 350
        self.height =  600
        
        #shape=(84,84,1)
        #self.shape=(shape[2], shape[0], shape[1])
        self.shape = (1, 84, 84)
    
    def reset(self) -> any:
        keyboard.press(' ') #game start
        keyboard.release(' ') #game start

        self.start_time = time.time()
        self.actual_image = self.getScreen()
        self.black_white_image = self.getWhiteBlackScreen()
        return self.black_white_image

    def close(self):
        print("ZAVÍRÁME:)")
    
    def step(self, action):

        if(action == 0.0):
            keyboard.press_and_release(' ')#up
        
        if(action == 1.0):#do nothing
            pass
        
        self.actual_image = self.getScreen()
        self.black_white_image = self.getWhiteBlackScreen()
        self.actual_reward = self.getReward()
        self.done = self.getDone()

        return self.black_white_image, self.actual_reward, self.done

    def getDone(self) -> bool:

        if self.black_white_image[34][33] == 0:
            return True
        return False

    def getReward(self) -> any:

        return (time.time() - self.start_time)

    def getWhiteBlackScreen(self) -> any:

        image = cv2.cvtColor(self.actual_image, cv2.COLOR_BGR2GRAY)
        (thresh, image) = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY)
        self.black_white_image = image
        return self.black_white_image

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

        new_frame = cv2.resize(img, self.shape[1:], interpolation=cv2.INTER_AREA)
        self.actual_image = new_frame

        return cv2.cvtColor(new_frame, cv2.COLOR_BGRA2RGB)