import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from environment import Environment
from replay import SimpleReplayBuffer

class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 512), 
            nn.ReLU(),
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(), 
            nn.Linear(128, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)
    
    def save_checkpoint(self, name_of_file):
        if not os.path.exists('models'):
            os.makedirs('models')
        print ("checkpoint saved...")
        torch.save(self.state_dict(), name_of_file)
    
    def load_checkpoint(self, name_of_file):
        print("checkpoint loaded...")
        self.load_state_dict(torch.load(name_of_file))

class DQNAgent:
    def __init__(
        self, 
        env: Environment,
        memory_size: int,
        batch_size: int,
        target_update: int,
        epsilon_decay: float =  1 / 20000,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.05,
        gamma: float = 0.99,
    ):
        obs_dim = env.simple_shape
        action_dim = env.simple_action_space.size
        
        self.env = env
        self.memory = SimpleReplayBuffer(obs_dim, memory_size, batch_size)
        self.batch_size = batch_size
        self.epsilon = max_epsilon
        self.epsilon_decay = epsilon_decay
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.target_update = target_update
        self.gamma = gamma
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
    
    def select_action(self, state: np.ndarray) -> np.ndarray:

        # epsilon greedy
        if self.epsilon > np.random.random():
            selected_action = np.random.choice(self.env.simple_action_space)
        else:
            selected_action = self.dqn(
                torch.FloatTensor(state).to(self.device)
            ).argmax()
            selected_action = selected_action.detach().cpu().numpy()
        
        self.transition = [state, selected_action]
        
        return selected_action
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, done = self.env.simple_step(action)

        if not self.is_test:
            self.transition += [reward, next_state, done]
            self.memory.store(*self.transition)
    
        return next_state, reward, done
    
    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch()

        loss = self._compute_dqn_loss(samples)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False
        scores = []
        i = 0
        update_cnt = 0
        while True:

            state = self.env.simple_reset()
            score = 0
            t0 = time.time()
            i = i + 1
            while True:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward

                # if training is ready
                if len(self.memory) >= self.batch_size:
                    self.loss = self.update_model()
                    update_cnt += 1
                    
                    # linearly decrease epsilon
                    self.epsilon = max(
                        self.min_epsilon, self.epsilon - (
                            self.max_epsilon - self.min_epsilon
                        ) * self.epsilon_decay
                    )
                    
                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()
                        print("Network was updated !")
                
                scores.append(score)
                # if episode ends
                if done:
                    scores.append(score)
                    break
            
            avg_score = np.mean(scores[-25:])
            t1 = time.time()
            t = t1-t0
            print('episode', i, 'last score %.0f, average score %.2f,  epsilon %.4f ' %
                (score, avg_score, self.epsilon),
                 'time ', t)

            with open ('logs/simple_last_score.txt', 'a') as fl:
                fl.write('%.0f\n' %(score))

            with open ('logs/simple_avg_score.txt', 'a') as fl:
                fl.write('%.3f \n' %(avg_score))
            
            with open ('logs/simple_times.txt', 'a') as fl:
                fl.write('%.3f \n' %(t))

            with open ('logs/simple_epsilon.txt', 'a') as fl:
                fl.write('%.4f\n' %(self.epsilon))
            
            with open ('logs/simple_loss.txt', 'a') as fl:
                fl.write('%.4f\n' %(self.loss))

            if i% 100 == 0:
                print("Saving model...")
                self.save(i)

    def save(self, epizode):
        
        self.checkpoint_dir = 'models/'
        self.checkpoint_file = os.path.join(self.checkpoint_dir,'MyDQN' + str(epizode))
        self.checkpoint_file_target = os.path.join(self.checkpoint_dir,'MyDQN_target' + str(epizode))
        self.dqn.save_checkpoint(self.checkpoint_file)
        self.dqn_target.save_checkpoint(self.checkpoint_file_target)
    
    def test(self) -> List[np.ndarray]:
        """Test the agent."""
        self.is_test = True
        
        state = self.env.reset()
        done = False
        score = 0
        
        frames = []
        while not done:
            frames.append(self.env.render(mode="rgb_array"))
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward
        
        print("score: ", score)
        self.env.close()
        
        return frames
    
    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.dqn_target(
            next_state
        ).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        # calculate dqn loss
        loss = F.smooth_l1_loss(curr_q_value, target)

        return loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    def _plot(
        self, 
        frame_idx: int, 
        scores: List[float], 
        losses: List[float], 
        epsilons: List[float],
    ):
        """Plot the training progresses."""
        plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. score: %s' % (frame_idx, np.mean(scores[-10:])))
        plt.plot(scores)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.subplot(133)
        plt.title('epsilons')
        plt.plot(epsilons)
        plt.show()