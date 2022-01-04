from environment import Environment
from fastai.vision.all import *
import keyboard
import cv2
import pathlib

def label_func(x): return x.parent.name


def load(path): # loads trained model
    learner = load_learner(path)
    print('Model loaded.')
    return(learner)
    
def run(learner):
    
    env = Environment()

    image = env.reset()

    while not keyboard.is_pressed("e"):
        
        cv2.imshow('Bot View', image)
        cv2.waitKey(1)

        result = learner.predict(image)
        action = result[0]

        if action == 'dup84':
            keyToPress = 0.0
        elif action == 'ddown84':
            keyToPress = 1.0
        else:
            keyToPress = 2.0

        image, reward, done =  env.step(keyToPress)

if __name__ == '__main__':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
<<<<<<< Updated upstream
    path = os.path.join('C:\SDU\AIG\project_master\jobData', 'model_resn18-5_84_150k.pkl')
=======
    path = os.path.join('C:\SDU\AIG\project_master\jobData', 'model_resn-5_84_150k.pkl')
>>>>>>> Stashed changes
    learner = load(path)
    run(learner)

