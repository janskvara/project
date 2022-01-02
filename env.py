import numpy as np
#import gym
import collections
import cv2


'''class RepeatActionAndMaxFrame(gym.Wrapper):

    def __init__(self, env=None, repeat=4):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2,self.shape), dtype=object)

    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        obs = self.env.reset()
        self.frame_buffer = np.zeros_like((2,self.shape), dtype=object)
        self.frame_buffer[0] = obs
        return obs

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape=(shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=np.float32(0), high=np.float32(1.0),
                                              shape=self.shape, dtype=np.float32)
    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)

        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = np.swapaxes(new_obs, 2,0)
        new_obs = new_obs / 255.0
        return new_obs

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             np.float32(env.observation_space.low.repeat(n_steps, axis=0)),
                             np.float32(env.observation_space.high.repeat(n_steps, axis=0)),
                             dtype=np.float32)
        self.stack = collections.deque(maxlen=n_steps)

    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        self.stack.append(observation)
        obs = np.array(self.stack).reshape(self.observation_space.low.shape)

        return obs

def make_env(env_name, shape=(84,84,1), skip=4):
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, skip)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, skip)

    return env'''


import cv2
import numpy as np
import win32gui, win32ui, win32con
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import pytesseract
import keyboard
import time

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
        #shape=(84,84,1)
        #self.shape=(shape[2], shape[0], shape[1])
        self.shape = (1, self.height, self.width)
        self._game_over_image = cv2.imread(r'jobData\restart.png')
    
    def reset(self) -> any:
        time.sleep(0.3)
        keyboard.press_and_release('space') #game start
        self.actual_image = self.getScreen()
        return self.getWhiteBlackScreen()

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
        self.black_white_image = self.getWhiteBlackScreen()
        self.actual_reward = self.getReward()
        self.done = self.getDone()

        return self.black_white_image, self.actual_reward, self.done

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
        new_frame = cv2.resize(img, (84,84), interpolation=cv2.INTER_AREA)

        return cv2.cvtColor(new_frame, cv2.COLOR_BGRA2RGB)
