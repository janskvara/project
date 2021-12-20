from grabScreen import grab_screen
from fastai.vision.all import *

def load(path):
    return(load_learner('learner.pkl'))

def decide(learner):
    _x_position = 0
    _y_position = 0
    width = 800
    height = 600
    image = grab_screen(region = (_x_position + 100, _y_position+300, width, 200))
    
    result = learner.predict(image)
    action = result[0]

    return action

if __name__ == '__main__':
    learner = load('learner.pkl')
    input = decide(learner)


