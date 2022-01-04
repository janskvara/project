import collections
import numpy as np
from typing import Dict


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.priorities = collections.deque(maxlen=max_size)


        self.state_memory = np.zeros((self.mem_size,*input_shape), dtype=np.float32)
        self.new_state_memory =np.zeros((self.mem_size,*input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size,dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.mem_size 
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.priorities.append(max(self.priorities, default=1))
        self.mem_cntr += 1
    #stochastic sampling method that interpolates between pure greedy prioritization and uniform random sampling (alpha determines how much prioritization is used)
    def get_probabilities(self):
        scaled_priorities = np.array(self.priorities)
        scaled_priorities = scaled_priorities/ scaled_priorities.sum()
        return scaled_priorities
    
    #Importance-sampling (IS) weight with Beta
    #For stability reasons, we always normalize weights by 1/max(importance) so that they only scale the update downwards.
    def get_importance(self, probabilities, beta):
        self.beta = beta
        importance =  np.power(1/self.mem_size * 1/probabilities, -self.beta)
        importance = importance / max(importance)
        return importance

    def sample(self, batch_size, beta):
        max_mem = min(self.mem_cntr, self.mem_size)
        sample_probs = self.get_probabilities()
        batch = np.random.choice(max_mem, batch_size, replace=False, p= sample_probs)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        importance = self.get_importance(sample_probs[batch], beta)

        return states, actions, rewards, next_states, dones, importance, batch
    
    #proportional prioritization
    def set_priorities(self, idx, errors, offset=1.1, alpha = 0.7):
            self.priorities[idx] = (np.abs(errors) + offset)** alpha


class SimpleReplayBuffer():

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size