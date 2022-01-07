import numpy as np
from environment import Environment
from typing import Dict, List, Tuple
import time

class RandomAgent:
    def __init__(
        self, 
        env: Environment,
    ):
        self.env = env

    def select_action(self) -> np.ndarray:
        # random choice
        selected_action = np.random.choice(self.env.simple_action_space)
        return selected_action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done = self.env.simple_step(action)

        return next_state, reward, done

    def test(self, number_of_epizodes):
        """Test the agent."""
        
        for epizode in range(number_of_epizodes):

            self.env.reset()
            scores = []
            score = 0
            t0 = time.time()
            while True:
                action = self.select_action()
                next_state, reward, done = self.step(action)
                score += reward

                scores.append(score)
                # if episode ends
                if done:
                    break
            t1 = time.time()
            avg_score = np.mean(scores[-25:])
            t = t1-t0

            print('episode', epizode, 'last score %.0f, average score %.2f' %
                (score, avg_score),
                'time ', t)

            with open ('logs/random_score.txt', 'a') as fl:
                fl.write('%.0f\n' %(score))

            with open ('logs/random_avg_score.txt', 'a') as fl:
                fl.write('%.3f \n' %(avg_score))
            
            with open ('logs/random_times.txt', 'a') as fl:
                fl.write('%.3f \n' %(t))

        self.env.close()
        return

    