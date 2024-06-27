import cv2
import numpy as np
import mss
import mss.tools
import pyautogui
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from collections import deque
import itertools
import time

pyautogui.PAUSE = 0
training_running = False
counter=0
# Define neural network architecture
class OsuManiaNet(nn.Module):
    def __init__(self, input_channels=1, key_state_size=4, output_size=8):
        super(OsuManiaNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self._to_linear = None
        self.convs(torch.randn(1, input_channels, 140, 50))
        self.fc1 = nn.Linear(self._to_linear + key_state_size, 256)
        self.fc2 = nn.Linear(256, output_size)

    def convs(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        if self._to_linear is None:
            self._to_linear = x.numel()
        return x

    def forward(self, x, key_states):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, key_states), dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 4, 2)
    
# Define reinforcement learning agent
class OsuManiaAgent:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = OsuManiaNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=4000)

    def select_action(self, state, key_states):
        with torch.no_grad():
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device) # Add batch and channel dimensions
            key_states = torch.tensor(np.array(key_states), dtype=torch.float32).unsqueeze(0).to(self.device)
            predictions = self.model(state, key_states)
            _, action_indices = torch.max(predictions, dim=2)
            action=action_indices.squeeze(0).tolist()
        return action

    def update_model(self, low, up):
        if up>len(self.replay_buffer):
            return
        batch=deque(itertools.islice(self.replay_buffer, low, up))
        states, key_states, actions = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        key_states = torch.tensor(np.array(key_states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(states, key_states)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, 2), actions.view(-1))
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes, env):
        global counter
        for episode in range(num_episodes):
            counter=0
            start=time.time()
            print("episode:",episode)
            if not training_running:
                break
            # Reset game state
            state, key_states = env.reset()
            done = False
            while not done and training_running:
                next_state, next_key_states, target_action, done = env.step()
                if done:
                    break
                self.replay_buffer.append((state, key_states, target_action))
                state = next_state
                key_states = next_key_states
        
            if done:
                end=time.time()
                print(end-start)
                print(counter)
                if counter/(end-start)>=27:
                    for i in range(8):
                        self.update_model(i*500,(i+1)*500)
                else:
                    self.replay_buffer.clear()
                
                time.sleep(4)
                self.save_model(f'Osu_ai/model/model_episode_{episode}.pth')
                pyautogui.click(1627,814)
                time.sleep(3.2)

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Model loaded from {path}')

# Define game environment
class OsuManiaEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(140, 50, 1), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([2, 2, 2, 2]) 
        self.key_mappings = {0: 'up', 1: 'down'}
        self.key_states = np.zeros(4)  # (D, F, J, K)
        self.keys = ['d', 'f', 'j', 'k']

    def reset(self):
        self.key_states = np.zeros(4)
        with mss.mss() as sct:
            rect=sct.monitors[0]
            img = np.array(sct.grab(rect))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        observation = self.preprocess_screen(img)
        return observation, self.key_states

    def step(self):
        global counter
        counter+=1
        with mss.mss() as sct:
            rect=sct.monitors[0]
            img = np.array(sct.grab(rect))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        observation = self.preprocess_screen(img[824:924, 300:580])
        target_action = self.extract_action(img[930:1080, 310:375], img[930:1080, 376:441], img[930:1080, 442:507], img[930:1080, 508:573])
        done = self.is_episode_done(img[0:140, 1320:1920])

        return observation, self.key_states, target_action, done

    def preprocess_screen(self, img):
        img = cv2.resize(img, (140, 50))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img

    def perform_action(self, action):
        for i, key_action in enumerate(action):
            if key_action == 1:  # Key down
                pyautogui.keyDown(self.keys[i])
                self.key_states[i] = 1
            elif key_action == 0:  # Key up
                pyautogui.keyUp(self.keys[i])
                self.key_states[i] = 0

    def extract_action(self, img1, img2, img3, img4):
        action=np.zeros(4)
        pink = cv2.imread('Osu_ai/pink.png')
        white = cv2.imread('Osu_ai/white.png')
        res = cv2.matchTemplate(img1, white, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= 0.1)
        if len(loc[0]) > 0:
            action[0]=1
        res = cv2.matchTemplate(img2, pink, cv2.TM_CCOEFF_NORMED) 
        loc = np.where(res >= 0.8)
        if len(loc[0]) > 0:
            action[1]=1
        res = cv2.matchTemplate(img3, pink, cv2.TM_CCOEFF_NORMED) 
        loc = np.where(res >= 0.8)
        if len(loc[0]) > 0:
            action[2]=1
        res = cv2.matchTemplate(img4, white, cv2.TM_CCOEFF_NORMED) 
        loc = np.where(res >= 0.2)
        if len(loc[0]) > 0:
            action[3]=1
        return action

    def is_episode_done(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        template = cv2.imread('Osu_ai/game_over.png', 0) 
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7  # Adjust this value as needed
        loc = np.where(res >= threshold)
        if len(loc[0]) > 0:
            return True
        else:
            return False    

def train_agent(agent, num_episodes):
    env = OsuManiaEnv()
    if agent==None:
        agent = OsuManiaAgent()
    agent.train(num_episodes, env)

