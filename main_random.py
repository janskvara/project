from random_agent import RandomAgent
from environment import Environment


env = Environment()
random_agent = RandomAgent(env)
random_agent.test(100)