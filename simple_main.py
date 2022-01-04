from environment import Environment
from myDQN import DQNAgent

env = Environment()

memory_size = 70000
batch_size = 64
target_update = 3000
epsilon_decay = 1 / 50000

agent = DQNAgent(env, memory_size, batch_size, target_update, epsilon_decay)
agent.train()