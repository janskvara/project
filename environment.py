import cv2
import numpy as np
import win32gui, win32ui, win32con
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

class Environment():

    _window_x_position = 0
    _window_y_position = 0
    _window_width = 800
    _window_height = 600

    _x_position = 0
    _y_position = 0
    _width = 0 
    _height = 0

    def __init__(self):

        self._browser_driver = webdriver.Chrome(ChromeDriverManager().install())
        self._browser_driver.set_window_rect(self._window_x_position, self._window_y_position, self._window_width, self._window_height)
        self._browser_driver.get("https://chromedino.com")

        """Ondra PC
        self._x_position = self._window_x_position + 50
        self._y_position = self._window_y_position + 240
        self._width =  self._window_width - 120
        self._height = 150
        """
        """Jenda PC
        self._x_position = self._window_x_position + 100
        self._y_position = self._window_y_position + 200
        self._width =  self._window_width - 50
        self._height =  200
        """
        # """Vašek PC
        self._x_position = self._window_x_position + 50
        self._y_position = self._window_y_position + 240
        self._width =  self._window_width - 120
        self._height = 150
        # """

    def getWhiteBlackScreen(self) -> any:

        image = self.getScreen()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (thresh, image) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return image

    def getScreen(self) -> any:

        hwin = win32gui.GetDesktopWindow()
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, self._width, self._height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (self._width, self._height), srcdc, (self._x_position,  self._y_position), win32con.SRCCOPY)
        
        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self._height, self._width, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)