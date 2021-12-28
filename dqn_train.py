from deep_reinforcement_learning_agent import DQNAgent
from environment import Environment

# parameters
num_frames = 200000000
memory_size = 2000
batch_size = 32
target_update = 200
epsilon_decay = 1 / 2000

env = Environment()

# train
agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
agent.train(num_frames)
agent.test()